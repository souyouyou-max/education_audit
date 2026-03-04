#!/bin/bash
# 本地开发启动脚本（热加载，读取 .env 环境变量）
set -a
source .env
set +a

uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8001 \
  --reload \
  --reload-dir app \
  --reload-dir static \
  --log-level info
