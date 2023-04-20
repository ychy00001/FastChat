#!/bin/bash
ps aux | grep "fastchat.serve" | grep -v grep | awk '{print $2}' |xargs kill -9