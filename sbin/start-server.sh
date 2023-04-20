#!/bin/bash
ps aux | grep "fastchat.serve.gradio_web_server" | grep -v grep | awk '{print $2}' |xargs kill -9

nohup python3 -m fastchat.serve.gradio_web_server --share --controller-url http://127.0.0.1:8081 --host 0.0.0.0 --port 8083 --model-list-mode reload >> /data/project/FastChat/log/gradio_web_server.out 2>&1 &