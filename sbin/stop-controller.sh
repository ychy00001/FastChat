#!/bin/bash
ps aux | grep "fastchat.serve.controller" | grep -v grep | awk '{print $2}' |xargs kill -9