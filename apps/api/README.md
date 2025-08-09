# AI Agent System - Backend API

基于FastAPI的AI智能体系统后端服务。

## 快速开始

### 安装依赖

```bash
uv sync
```

### 配置环境变量

```bash
cp ../../.env.example .env
# 编辑.env文件中的配置
```

### 运行开发服务器

```bash
uv run python src/main.py
```

或者使用uvicorn:

```bash
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### API文档

启动服务后访问:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 开发

### 代码质量检查

```bash
# 格式化代码
uv run black src tests

# 代码检查
uv run ruff check src tests

# 类型检查
uv run mypy src
```

### 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行测试并生成覆盖率报告
uv run pytest --cov=src --cov-report=html
```

## 项目结构

- `src/` - 源代码
  - `api/` - API路由层
  - `core/` - 核心配置和工具
  - `services/` - 业务逻辑层
  - `models/` - 数据模型
  - `ai/` - AI集成模块
- `tests/` - 测试代码