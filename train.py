import torch
from s3dp import S3DP
from config import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import CONFIG_MAPPING
from forward_backward import forward_backward_pipelining_without_interleaving
from gpt import forward_step
from dataset import TestDataset
from torch.utils.data import DataLoader
from gpt import forward_step, GPTConfig
import os 
import torch.distributed as dist


if __name__ == "__main__":
    # model_path = "/shared/public/models/models-do-not-use-for-production/llama-2-7b-hf/"
    # model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)

    os.environ['NCCL_DEBUG'] = 'WARN'
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    hf_model = AutoModelForCausalLM.from_config(CONFIG_MAPPING["gpt2"](), trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = S3DP(hf_model, config)
    tokenizer = AutoTokenizer.from_pretrained(GPTConfig.model_path)
    tokenizer.pad_token_id = hf_model.config.eos_token_id
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    dataset = TestDataset(tokenizer)
    data_loader = DataLoader(dataset, batch_size=config.micro_batch_size, num_workers=8, pin_memory=True)
    data_iterator = iter(data_loader)
    
    losses_reduced = forward_backward_pipelining_without_interleaving(
        forward_step_func=forward_step,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=config.micro_batch_size,
        forward_only=True,
        collect_non_loss_data=False,
        pipeline_model_parallel_world_size=config.pipeline_model_parallel_size,
        pipeline_model_parallel_rank=model.local_rank_in_pipeline,
        dtype=config.dtype,
        group=model.pipeline_model_parallel_group,
        tensor_shape=[1, dataset.max_length, hf_model.config.n_embd],
        prev_rank=model.prev_rank_in_pipeline,
        next_rank=model.next_rank_in_pipeline
        )
    