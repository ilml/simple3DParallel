# simple3DParallel

S3DP(simple 3d parallel) aims at implementing data parallel, tensor parallel and pipeline parallel for efficient distributed LLM training.
Inspired by Megatron-LM, we want to provide a simple/minimal framework for people with no experience on distribued training so that they can understand it's basics.

#### Features of S3DP
- Sticking to native PyTorch and Huggingface APIs as much as possible.
- Simple and efficient implementation of only the core functions of 3d parallelism.

#### Getting started

just run `sh train.sh`

This repo is WIP and for study use only.

## Development Log
2023/11/13 -- Model sharding ready, tested GPT2 on 8 GPUs.
2023/12/16 -- Pipeline parallelism forward pass ready. Tested GPT2 model on V100 node(6 gpus) with pp size=6(since GPT2 has 12 blocks, placing 2 blocks on each gpu).
