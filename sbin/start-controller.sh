#!/bin/bash
ps aux | grep "fastchat.serve.controller" | grep -v grep | awk '{print $2}' |xargs kill -9

nohup python3 -m fastchat.serve.controller --host 0.0.0.0 --port 8081 >> /data/project/FastChat/log/controller.out 2>&1 &
