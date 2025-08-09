#!/bin/bash

# AI Agent System - 开发环境设置脚本
set -e

echo "🚀 Setting up AI Agent System development environment..."

# 检查Docker是否安装
if ! command -v docker >/dev/null 2>&1; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose >/dev/null 2>&1; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# 检查uv是否安装
if ! command -v uv >/dev/null 2>&1; then
    echo "❌ uv is not installed. Please install uv first."
    echo "Run: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "📁 Project root: $PROJECT_ROOT"

# 进入项目根目录
cd "$PROJECT_ROOT"

# 创建环境变量文件
if [ ! -f .env ]; then
    echo "📄 Creating .env file from template..."
    cp .env.example .env
    echo "✅ Please edit .env file with your configuration"
fi

# 启动数据库服务
echo "🐳 Starting database services..."
docker compose -f infrastructure/docker/docker-compose.yml up -d postgres redis qdrant

# 等待数据库启动
echo "⏳ Waiting for databases to be ready..."
sleep 10

# 检查数据库连接
echo "🔍 Checking database connections..."
docker compose -f infrastructure/docker/docker-compose.yml exec -T postgres pg_isready -U ai_agent_user -d ai_agent_db || true

# 设置API环境
echo "🐍 Setting up Python API environment..."
cd apps/api

# 安装依赖
uv sync

echo "🧪 Running tests to verify setup..."
cd "$PROJECT_ROOT"
bash scripts/test-api.sh

echo "✅ Development environment setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your API keys and configuration"
echo "2. Start the API server: npm run api:dev"
echo "3. Visit API docs: http://localhost:8000/docs"
echo "4. PgAdmin (optional): docker compose --profile tools up -d"
echo "   - URL: http://localhost:8080"
echo "   - Email: admin@aiagent.com"
echo "   - Password: admin123"
echo ""
echo "🐳 Docker services:"
echo "- PostgreSQL: localhost:5432"
echo "- Redis: localhost:6379"
echo "- Qdrant: localhost:6333"