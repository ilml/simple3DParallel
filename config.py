from dataclasses import dataclass
import torch


@dataclass
class config:
    world_size = 6
    tensor_model_parallel_size = 1 
    pipeline_model_parallel_size = 6
    micro_batch_size = 12
    dtype = torch.bfloat16
    
    