import torch
from s3dp import S3DP
from config import config
from transformers import AutoModelForCausalLM
from transformers import CONFIG_MAPPING

if __name__ == "__main__":
    # model_path = "/shared/public/models/models-do-not-use-for-production/llama-2-7b-hf/"
    # model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)

    
    model = AutoModelForCausalLM.from_config(CONFIG_MAPPING["gpt2"](), trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = S3DP(model, config)
    