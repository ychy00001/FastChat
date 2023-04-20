#!/bin/bash
ps aux | grep "fastchat.serve.gradio_web_server" | grep -v grep | awk '{print $2}' |xargs kill -9
