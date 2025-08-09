#!/bin/bash

# API测试脚本
set -e

echo "🧪 Running API tests..."

# 进入API目录
cd "$(dirname "$0")/../apps/api"

# 设置Python路径
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# 复制测试环境变量
cp .env.test .env

# 运行测试
uv run pytest tests/ -v

echo "✅ API tests completed successfully!"