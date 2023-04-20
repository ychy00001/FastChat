#!/bin/bash
ps aux | grep "fastchat.serve.model_worker" | grep "1200step" | grep -v grep | awk '{print $2}' |xargs kill -9
