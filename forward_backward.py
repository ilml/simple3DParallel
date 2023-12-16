from communication import *
import torch
from utils import deallocate_output_tensor, get_attr_wrapped_model
from typing import Callable, Iterator, List, Optional, Union
from gpt import loss_func

def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    pipeline_model_parallel_world_size: int = 1,
    pipeline_model_parallel_rank: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    group: torch.distributed.ProcessGroup,
    tensor_shape: List[int],
    prev_rank: int,
    next_rank: int
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    is_first_stage = pipeline_model_parallel_rank == 0
    is_last_stage = pipeline_model_parallel_rank == pipeline_model_parallel_world_size - 1

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        pipeline_model_parallel_world_size
        - pipeline_model_parallel_rank
        - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        input_tensor = recv_forward(tensor_shape, prev_rank, next_rank, dtype, group, is_first_stage)
        output_tensor = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            collect_non_loss_data,
            is_first_stage,
            is_last_stage
        )
        send_forward(output_tensor[0], prev_rank, next_rank, dtype, group, is_last_stage)

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor, True)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = recv_forward(tensor_shape, prev_rank, next_rank, dtype, group, is_first_stage)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        output_tensor = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            collect_non_loss_data,
            is_first_stage,
            is_last_stage
        )

        if forward_only:
            if not is_last_stage:
                send_forward(output_tensor[0], prev_rank, next_rank, dtype, group, is_last_stage)

            if not last_iteration:
                input_tensor = recv_forward(tensor_shape, prev_rank, next_rank, dtype, group, is_first_stage)

        else:
            output_tensor_grad = send_forward_recv_backward(
                output_tensor[0], prev_rank, next_rank, tensor_shape, dtype, group, is_last_stage
            )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], True)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # backward pass completes in the 1F1B stage.
            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad
            )

            if last_iteration:
                input_tensor = None
                send_backward(input_tensor_grad, prev_rank, next_rank, dtype, group, is_first_stage)
            else:
                input_tensor = send_backward_recv_forward(
                    input_tensor_grad, prev_rank, next_rank, tensor_shape, dtype, group, is_first_stage
                )

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = recv_backward(tensor_shape, prev_rank, next_rank, dtype, group, is_last_stage)

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad 
            )
            send_backward(input_tensor_grad, dtype, group, is_first_stage)


    """
    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func([model])
    """

    return forward_data_store


def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    collect_non_loss_data,
    is_first_stage, 
    is_last_stage,
):
    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""

    if is_first_stage:
        output_tensor = forward_step_func(data_iterator, model)
    elif is_last_stage:
        output_tensor = model.post_process_layer(input_tensor)
        if not collect_non_loss_data:
            output_tensor = loss_func(output_tensor)
            loss, loss_reduced = output_tensor
            output_tensor = loss / num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)
    else:
        input_tensor = [input_tensor]
        for layer in model.layers:
            input_tensor = layer(input_tensor[0])
        output_tensor = input_tensor
    return output_tensor
 
 
    
def backward_step(input_tensor, output_tensor, output_tensor_grad):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""


    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    return input_tensor_grad
