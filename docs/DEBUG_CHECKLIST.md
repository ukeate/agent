# AI Agent系统调试检查清单

## 🔍 问题排查步骤

### 1. 服务启动问题

**症状**: 服务无法启动或访问
```bash
# 检查端口占用
lsof -i :8000  # 后端端口
lsof -i :3000  # 前端端口

# 检查进程状态
ps aux | grep uvicorn
ps aux | grep node

# 重启服务
pkill -f uvicorn
pkill -f "node.*vite"
```

### 2. 前端页面空白

**症状**: 浏览器显示空白页面
```bash
# 检查控制台错误
# 打开浏览器开发者工具 -> Console

# 检查网络请求
# 打开浏览器开发者工具 -> Network

# 检查前端编译错误
npm run build
```

### 3. API请求失败

**症状**: 前端无法连接后端API
```bash
# 检查后端服务状态
curl http://localhost:8000/api/v1/agent/status

# 检查前端代理配置
cat apps/web/vite.config.ts | grep proxy -A 10

# 检查CORS配置
curl -H "Origin: http://localhost:3000" \
     -H "Access-Control-Request-Method: POST" \
     http://localhost:8000/api/v1/agent/chat
```

### 4. 数据库连接问题

**症状**: 数据库相关错误
```bash
# 检查Docker容器状态
docker ps | grep postgres
docker ps | grep redis
docker ps | grep qdrant

# 检查数据库连接
docker exec -it infrastructure-postgres-1 pg_isready

# 重启数据库服务
cd infrastructure/docker
docker-compose restart postgres redis qdrant
```

### 5. AutoGen功能异常

**症状**: 多智能体对话创建失败
```bash
# 检查AutoGen配置
python -c "from ai.autogen import create_default_agents; print(create_default_agents())"

# 检查OpenAI API配置
echo $OPENAI_API_KEY

# 运行AutoGen测试
cd apps/api
python -m pytest tests/ai/autogen/ -v
```

## 🛠️ 调试工具使用

### VS Code调试配置

在`.vscode/launch.json`中添加：
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug FastAPI",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/apps/api/src/main.py",
            "console": "integratedTerminal",
            "args": ["--reload", "--port", "8000"]
        },
        {
            "name": "Debug React",
            "type": "node",
            "request": "launch",
            "cwd": "${workspaceFolder}/apps/web",
            "runtimeExecutable": "npm",
            "runtimeArgs": ["run", "dev"]
        }
    ]
}
```

### Chrome DevTools

1. **React组件调试**
   - 安装React Developer Tools扩展
   - 查看组件状态和props
   - 使用Profiler分析性能

2. **网络请求调试**
   - Network标签查看API请求
   - 检查请求头和响应数据
   - 分析加载时间

3. **控制台调试**
   - 查看JavaScript错误
   - 使用console.log输出调试信息
   - 使用断点调试

## 📊 性能调试

### 前端性能
```bash
# 分析打包大小
npm run build
npm run analyze

# 检查内存泄漏
# 使用Chrome DevTools -> Memory标签

# 性能分析
# 使用Chrome DevTools -> Performance标签
```

### 后端性能
```bash
# 使用性能分析工具
pip install py-spy
py-spy top --pid <uvicorn_pid>

# 数据库性能监控
docker exec -it infrastructure-postgres-1 \
  psql -U postgres -c "SELECT * FROM pg_stat_activity;"
```

## 🔧 故障恢复步骤

### 完全重启系统
```bash
# 1. 停止所有服务
pkill -f uvicorn
pkill -f "node.*vite"
cd infrastructure/docker && docker-compose down

# 2. 清理缓存
cd apps/web && rm -rf node_modules/.vite
cd apps/api && rm -rf .pytest_cache __pycache__

# 3. 重新启动
cd infrastructure/docker && docker-compose up -d
cd apps/api/src && python -m uvicorn main:app --reload &
cd apps/web && npm run dev &
```

### 数据重置
```bash
# 重置数据库
cd infrastructure/docker
docker-compose down -v
docker-compose up -d

# 重新初始化数据
cd apps/api
python -c "from core.database import init_db; init_db()"
```

## 📝 调试日志

### 启用详细日志
```python
# 在apps/api/src/core/logging.py中设置
import structlog
import logging

logging.basicConfig(level=logging.DEBUG)
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
)
```

### 查看系统日志
```bash
# 应用日志
tail -f /tmp/uvicorn.log
tail -f /tmp/vite.log

# 系统日志
journalctl -f -u docker
```

## 🚨 紧急问题联系

如遇到无法解决的问题：
1. 收集错误日志和堆栈跟踪
2. 记录重现步骤
3. 检查环境变量和配置文件
4. 运行系统诊断脚本: `./scripts/debug-system.sh`