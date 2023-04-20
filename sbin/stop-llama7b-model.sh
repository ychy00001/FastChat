#!/bin/bash
ps aux | grep "fastchat.serve.model_worker" | grep "llama-7b-hf" | grep -v grep | awk '{print $2}' |xargs kill -9
