#!/bin/bash
ps aux | grep "fastchat.serve.model_worker" | grep "llama-sft-1200" | grep -v grep | awk '{print $2}' |xargs kill -9

nohup python3 -m fastchat.serve.model_worker --model-path /data/model/llama-sft-1200 --host 127.0.0.1 --port 8082 --worker-address http://127.0.0.1:8082 --controller-address http://127.0.0.1:8081 --num-gpus 4 >> /data/project/FastChat/log/model_worker.out 2>&1 &
