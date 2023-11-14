from dataclasses import dataclass


@dataclass
class config:
    world_size = 8
    tensor_model_parallel_size = 1 
    pipeline_model_parallel_size = 4
    