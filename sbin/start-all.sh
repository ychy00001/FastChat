#!/bin/bash
ps aux | grep "fastchat.serve" | grep -v grep | awk '{print $2}' |xargs kill -9

nohup python3 -m fastchat.serve.controller --host 0.0.0.0 --port 8081 >> /data/project/FastChat/log/controller.out 2>&1 &

# llama-7b
nohup python3 -m fastchat.serve.model_worker --model-path /data/model/llama-7b-hf --host 127.0.0.1 --port 8082 --worker-address http://127.0.0.1:8082 --controller-address http://127.0.0.1:8081 --num-gpus 4 >> /data/project/FastChat/log/model_worker.out 2>&1 &

# bloom-7b
#python3 -m fastchat.serve.model_worker --model-path /data/model/bloomz-7b1-mt --host 127.0.0.1 --port 8082 --worker-address http://127.0.0.1:8082 --model-name bloomz-7b1-mt  --controller-address http://127.0.0.1:8081 --num-gpus 4 >> /data/project/FastChat/log/model_worker.out 2>&1 &

nohup python3 -m fastchat.serve.gradio_web_server --concurrency-count 30 --controller-url http://127.0.0.1:8081 --host 0.0.0.0 --port 8083 --model-list-mode reload >> /data/project/FastChat/log/gradio_web_server.out 2>&1 &