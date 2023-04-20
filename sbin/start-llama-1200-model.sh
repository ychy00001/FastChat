#!/bin/bash
ps aux | grep "fastchat.serve.model_worker" | grep "1200step" | grep -v grep | awk '{print $2}' |xargs kill -9

nohup python3 -m fastchat.serve.model_worker  --max-gpu-memory "75GiB" --limit-model-concurrency 10 --model-name "从容V0.1.4" --model-path /data/model/1200step --host 127.0.0.1 --port 8082 --worker-address http://127.0.0.1:8082 --controller-address http://127.0.0.1:8081 --num-gpus 4 >> /data/project/FastChat/log/model_worker.out 2>&1 &
