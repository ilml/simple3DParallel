# simple3DParallel

S3DP(simple 3d parallel) aims at implementing data parallel, tensor parallel and pipeline parallel for efficient distributed LLM training.
Inspired by Megatron-LM, we want to provide a simple/minimal framework for people with no experience on distribued training so that they can understand its basics.

#### Features of S3DP
- Sticking to native PyTorch and Huggingface APIs as much as possible.
- Simple and efficient implementation of only the core functions of 3d parallelism.


This repo is WIP and for study use only.

## Development Log
2023/11/13 -- Model sharding ready, tested GPT2 on 8 GPUs.

| Model size | TP | PP | DP | Model FLOPs Utilization |
|:----------:|:-----------:|:-----------:|:-----------:|:-----------:|
|    124M     |         |        |         |       |
