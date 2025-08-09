#!/bin/bash

echo "🔧 AI Agent系统调试脚本"
echo "========================="

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查服务状态
echo -e "${YELLOW}1. 检查基础服务状态${NC}"
echo "检查PostgreSQL..."
if docker ps | grep -q postgres; then
    echo -e "${GREEN}✅ PostgreSQL运行中${NC}"
else
    echo -e "${RED}❌ PostgreSQL未运行${NC}"
fi

echo "检查Redis..."
if docker ps | grep -q redis; then
    echo -e "${GREEN}✅ Redis运行中${NC}"
else
    echo -e "${RED}❌ Redis未运行${NC}"
fi

echo "检查Qdrant..."
if docker ps | grep -q qdrant; then
    echo -e "${GREEN}✅ Qdrant运行中${NC}"
else
    echo -e "${RED}❌ Qdrant未运行${NC}"
fi

echo -e "\n${YELLOW}2. 检查应用服务状态${NC}"
echo "检查后端API服务..."
if curl -s http://localhost:8000/api/v1/agent/status > /dev/null; then
    echo -e "${GREEN}✅ 后端API服务正常${NC}"
    curl -s http://localhost:8000/api/v1/agent/status | jq '.data.health'
else
    echo -e "${RED}❌ 后端API服务异常${NC}"
fi

echo "检查前端服务..."
if curl -s -I http://localhost:3000/ | grep -q "200 OK"; then
    echo -e "${GREEN}✅ 前端服务正常${NC}"
else
    echo -e "${RED}❌ 前端服务异常${NC}"
fi

echo -e "\n${YELLOW}3. 检查多智能体功能${NC}"
echo "多智能体健康检查..."
HEALTH_RESPONSE=$(curl -s http://localhost:8000/api/v1/multi-agent/health)
if echo "$HEALTH_RESPONSE" | jq -e '.healthy' > /dev/null; then
    echo -e "${GREEN}✅ 多智能体服务健康${NC}"
    echo "$HEALTH_RESPONSE" | jq '.service_info'
else
    echo -e "${RED}❌ 多智能体服务异常${NC}"
fi

echo -e "\n${YELLOW}4. 测试API接口${NC}"
echo "测试单智能体对话..."
CHAT_RESPONSE=$(curl -s -X POST "http://localhost:8000/api/v1/agent/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "系统调试测试", "stream": false}')

if echo "$CHAT_RESPONSE" | jq -e '.success' > /dev/null; then
    echo -e "${GREEN}✅ 单智能体对话功能正常${NC}"
else
    echo -e "${RED}❌ 单智能体对话功能异常${NC}"
    echo "响应: $CHAT_RESPONSE"
fi

echo -e "\n${YELLOW}5. 检查前后端连通性${NC}"
echo "通过前端代理测试API..."
PROXY_RESPONSE=$(curl -s -X POST "http://localhost:3000/api/v1/agent/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "前端代理测试", "stream": false}')

if echo "$PROXY_RESPONSE" | jq -e '.success' > /dev/null; then
    echo -e "${GREEN}✅ 前端API代理正常${NC}"
else
    echo -e "${RED}❌ 前端API代理异常${NC}"
fi

echo -e "\n${YELLOW}6. 检查页面路由${NC}"
if curl -s -I http://localhost:3000/multi-agent | grep -q "200 OK"; then
    echo -e "${GREEN}✅ 多智能体页面路由正常${NC}"
else
    echo -e "${RED}❌ 多智能体页面路由异常${NC}"  
fi

echo -e "\n${YELLOW}7. 系统资源状态${NC}"
echo "内存使用情况:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -4

echo -e "\n${GREEN}调试脚本执行完成！${NC}"
echo "如需查看详细日志："
echo "  后端日志: tail -f /tmp/uvicorn.log"
echo "  前端日志: tail -f /tmp/vite.log"
echo "  Docker日志: cd infrastructure/docker && docker-compose logs -f"