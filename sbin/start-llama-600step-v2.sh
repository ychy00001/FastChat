#!/bin/bash
ps aux | grep "fastchat.serve.model_worker" | grep "llama_hf_65B_600step_v2" | grep -v grep | awk '{print $2}' |xargs kill -9

export export CUDA_VISIBLE_DEVICES=0,1,2,3 && nohup python3 -m fastchat.serve.model_worker  --max-gpu-memory "78GiB" --limit-model-concurrency 10 --model-path /data/model/llama_hf_65B_600step_v2 --host 127.0.0.1 --port 8090 --worker-address http://127.0.0.1:8090 --controller-address http://127.0.0.1:8081 --num-gpus 4 >> /data/project/FastChat/log/model_worker.out 2>&1 &