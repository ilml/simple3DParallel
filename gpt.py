from functools import partial
import torch
from torch import Tensor
import os

class GPTConfig:
    model_path = "/shared/public/models/gpt2"

def forward_step(data_iterator, model):
    """Forward training step.
    """
    input = next(data_iterator)
    # we ignore attention mask for debug now
    input = input["input_ids"][0].to("cuda:{}".format(model.local_rank))
    input = model.pre_process_layer(input)
    input = input.unsqueeze(0)
    for layer in model.layers:
        input = layer(input[0])
    return input

    
def loss_func(output_tensor: Tensor):
    """Loss function.

    Args:
        loss_mask (Tensor): Used to mask out some portions of the loss
        output_tensor (Tensor): The tensor with the losses
    """    

    losses = output_tensor.float()
    #loss_mask = loss_mask.view(-1).float()
    #loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    loss = torch.sum(losses) 

    # Check individual rank losses are not NaN prior to DP all-reduce.
    global_rank = torch.distributed.get_rank()
    assert not loss.isnan(), (
        f'Rank {global_rank}: found NaN in local forward loss calculation. '
        f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
    )
    print("loss:", loss)
    return loss , {'lm loss': loss}

    