"""
Communication utilities for S3DP.
"""
import torch
from typing import Callable, List, Optional, Tuple, Union

# Types
Shape = Union[List[int], torch.Size]

def _batched_p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    prev_rank: int,
    next_rank: int,
    group: torch.distributed.ProcessGroup
    ):
    ops = []
    if tensor_send_prev is not None:
        send_prev_op = torch.distributed.P2POp(
            torch.distributed.isend,
            tensor_send_prev,
            prev_rank,
            group,
        )
        ops.append(send_prev_op)
    if tensor_recv_prev is not None:
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            tensor_recv_prev,
            prev_rank,
            group,
        )
        ops.append(recv_prev_op)
    if tensor_send_next is not None:
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend,
            tensor_send_next,
            next_rank,
            group,
        )
        ops.append(send_next_op)
    if tensor_recv_next is not None:
        recv_next_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            tensor_recv_next,
            next_rank,
            group,
        )
        ops.append(recv_next_op)

    reqs = torch.distributed.batch_isend_irecv(ops) if len(ops) else []
    return reqs

    
def _communicate(
    *,
    tensor_send_next: Optional[torch.Tensor],
    tensor_send_prev: Optional[torch.Tensor],
    recv_prev: bool,
    recv_next: bool,
    prev_rank: int,
    next_rank: int,
    tensor_shape: Shape,
    dtype: torch.dtype, 
    group: torch.distributed.ProcessGroup 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Communicate tensors between stages. Used as helper method in other
    communication methods that are used in megatron/schedules.py.

    Arguments:
        tensor_send_next (torch.Tensor, optional):
            Tensor to send to next rank (no tensor sent if None)

        tensor_send_prev (torch.Tensor, optional):
            Tensor to send to prev rank (no tensor sent if None)

        recv_prev (boolean, required):
            whether tensor should be received from previous rank.

        recv_next (boolean, required):
            whether tensor should be received from next rank.

        tensor_shape (List[int] or torch.Size, required):
            shape of tensor to receive (this method assumes that all
            tensors sent and received in a single function call are
            the same shape).

        wait_on_reqs (boolean, optional, default=False):
            For non-batched p2p communication, wait on each request
            before returning.

    Returns:
        tuple containing

        - tensor_recv_prev: torch.Tensor if recv_prev is True, None otherwise.
        - tensor_recv_next: torch.Tensor if recv_next is True, None otherwise.

    """

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None

    if recv_prev:
        tensor_recv_prev = torch.empty(
            tensor_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=dtype,
        )
    if recv_next:
        tensor_recv_next = torch.empty(
            tensor_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=dtype,
        )
    
    reqs = _batched_p2p_ops(
        tensor_send_prev=tensor_send_prev,
        tensor_recv_prev=tensor_recv_prev,
        tensor_send_next=tensor_send_next,
        tensor_recv_next=tensor_recv_next,
        prev_rank=prev_rank,
        next_rank=next_rank,
        group=group,
    )

    if len(reqs) > 0:
        for req in reqs:
            req.wait()
        reqs = None

    torch.cuda.synchronize()

    return tensor_recv_prev, tensor_recv_next, reqs

    
def recv_forward(tensor_shape: Shape, 
                 prev_rank: int,
                 next_rank: int,
                 dtype: torch.dtype, 
                 group: torch.distributed.ProcessGroup, 
                 is_first_stage: bool
                 ) -> torch.Tensor:
    """ Receive tensor from previous rank in pipeline (forward receive).


    See _communicate for argument details.
    """

    if not is_first_stage:
        input_tensor, _, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
            prev_rank=prev_rank,
            next_rank=next_rank,    
            tensor_shape=tensor_shape,
            dtype=dtype,
            group=group
        )
    else:
        input_tensor = None
    return input_tensor


def recv_backward(tensor_shape: Shape, 
                 prev_rank: int,
                 next_rank: int,
                 dtype: torch.dtype, 
                 group: torch.distributed.ProcessGroup, 
                 is_last_stage: bool
                  ) -> torch.Tensor:
    """Receive tensor from next rank in pipeline (backward receive).

    See _communicate for argument details.
    """
    if not is_last_stage:
        _, output_tensor_grad, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            prev_rank=prev_rank,
            next_rank=next_rank,    
            tensor_shape=tensor_shape,
            dtype=dtype,
            group=group
        )
    else:
        output_tensor_grad = None
    return output_tensor_grad
    

def send_forward(output_tensor: torch.Tensor,
                 prev_rank: int,
                 next_rank: int,
                 dtype: torch.dtype, 
                 group: torch.distributed.ProcessGroup, 
                 is_last_stage: bool
                 ) -> None:
    """Send tensor to next rank in pipeline (forward send).

    See _communicate for argument details.
    """

    if not is_last_stage:
        _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            prev_rank=prev_rank,
            next_rank=next_rank,    
            tensor_shape=None,
            dtype=dtype,
            group=group
        )


def send_backward(input_tensor_grad: torch.Tensor, 
                 prev_rank: int,
                 next_rank: int,
                 dtype: torch.dtype, 
                 group: torch.distributed.ProcessGroup, 
                 is_first_stage: bool 
                 ) -> None:
    """Send tensor to previous rank in pipeline (backward send).

    See _communicate for argument details.
    """
    if not is_first_stage:
        _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=False,
            prev_rank=prev_rank,
            next_rank=next_rank,    
            tensor_shape=None,
            dtype=dtype,
            group=group
        )
        

def send_forward_recv_backward(output_tensor: torch.Tensor, 
                               prev_rank: int,
                               next_rank: int,
                               tensor_shape: Shape, 
                               dtype: torch.dtype, 
                               group: torch.distributed.ProcessGroup, 
                               is_last_stage: bool 
                               ) -> torch.Tensor:
    """Batched send and recv with next rank in pipeline.

    See _communicate for argument details.
    """
    if is_last_stage:
        output_tensor_grad = None
    else:
        _, output_tensor_grad, _ = _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            prev_rank=prev_rank,
            next_rank=next_rank,    
            tensor_shape=tensor_shape,
            dtype=dtype,
            group=group
        )
    return output_tensor_grad


def send_backward_recv_forward(input_tensor_grad: torch.Tensor, 
                               prev_rank: int,
                               next_rank: int,
                               tensor_shape: Shape, 
                               dtype: torch.dtype, 
                               group: torch.distributed.ProcessGroup, 
                               is_first_stage: bool 
                               ) -> torch.Tensor:
    """Batched send and recv with previous rank in pipeline.

    See _communicate for argument details.
    """
    if is_first_stage:
        input_tensor = None
    else:
        input_tensor, _, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=True,
            recv_next=False,
            prev_rank=prev_rank,
            next_rank=next_rank,    
            tensor_shape=tensor_shape,
            dtype=dtype,
            group=group
        )
    return input_tensor