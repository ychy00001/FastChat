#!/bin/bash
ps aux | grep "fastchat.serve.model_worker" | grep "llama-sft-1200" | grep -v grep | awk '{print $2}' |xargs kill -9
