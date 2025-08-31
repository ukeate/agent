#!/bin/bash

# NATS集群启动脚本
set -e

# 配置变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.nats.yml"
ENV_FILE="${SCRIPT_DIR}/.env"

echo "🚀 启动NATS JetStream集群..."

# 创建环境文件（如果不存在）
if [[ ! -f "$ENV_FILE" ]]; then
    echo "📝 创建环境配置文件..."
    cat > "$ENV_FILE" << 'EOF'
# NATS集群配置
NATS_PASSWORD=s3cr3t_nats_password
CLUSTER_PASSWORD=cluster_s3cr3t

# JetStream配置
JETSTREAM_MAX_MEMORY=2GB
JETSTREAM_MAX_FILE=20GB

# 日志级别
LOG_LEVEL=info
DEBUG_MODE=false
EOF
    echo "✅ 环境文件已创建: $ENV_FILE"
fi

# 确保数据目录存在且权限正确
echo "📁 检查数据目录..."
for node in nats-1 nats-2 nats-3; do
    mkdir -p "${SCRIPT_DIR}/data/${node}/jetstream"
    mkdir -p "${SCRIPT_DIR}/data/${node}/logs"
    
    # 设置目录权限
    chmod 755 "${SCRIPT_DIR}/data/${node}"
    chmod 755 "${SCRIPT_DIR}/data/${node}/jetstream"
    chmod 755 "${SCRIPT_DIR}/data/${node}/logs"
done

# 停止现有的集群（如果运行中）
echo "🛑 停止现有的NATS集群（如果存在）..."
docker-compose -f "$COMPOSE_FILE" down --volumes --remove-orphans 2>/dev/null || true

# 清理旧的容器和网络
echo "🧹 清理Docker资源..."
docker container prune -f 2>/dev/null || true
docker network prune -f 2>/dev/null || true

# 启动NATS集群
echo "🏁 启动NATS集群..."
docker-compose -f "$COMPOSE_FILE" up -d

# 等待服务启动
echo "⏳ 等待NATS节点启动..."
sleep 30

# 检查集群状态
echo "🔍 检查集群状态..."

# 检查每个节点的健康状态
for i in {1..3}; do
    port=$((4221 + i))
    node="nats-$i"
    
    echo "检查节点 $node (端口: $port)..."
    
    # 等待节点启动
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$((8221 + i))/healthz" | grep -q "200"; then
            echo "✅ $node 健康检查通过"
            break
        else
            echo "⏳ 等待 $node 启动... (尝试 $attempt/$max_attempts)"
            sleep 2
            ((attempt++))
        fi
        
        if [ $attempt -gt $max_attempts ]; then
            echo "❌ $node 启动超时"
            exit 1
        fi
    done
done

# 显示集群信息
echo ""
echo "🎉 NATS集群启动成功！"
echo ""
echo "📊 集群信息:"
echo "  - 集群名称: agent-cluster"
echo "  - 节点数量: 3"
echo "  - JetStream: 已启用"
echo ""
echo "🌐 连接信息:"
echo "  - NATS-1: localhost:4222 (监控: http://localhost:8222)"
echo "  - NATS-2: localhost:4223 (监控: http://localhost:8223)"  
echo "  - NATS-3: localhost:4224 (监控: http://localhost:8224)"
echo "  - 集群监控: http://localhost:7777"
echo ""
echo "🔑 认证信息:"
echo "  - 用户名: agent_system"
echo "  - 密码: 请查看配置文件"
echo ""
echo "💡 有用的命令:"
echo "  - 查看日志: docker-compose -f $COMPOSE_FILE logs -f"
echo "  - 停止集群: docker-compose -f $COMPOSE_FILE down"
echo "  - 重启集群: docker-compose -f $COMPOSE_FILE restart"
echo ""

# 可选：检查JetStream状态
if command -v nats &> /dev/null; then
    echo "🔍 JetStream状态检查:"
    echo "  运行以下命令查看JetStream信息:"
    echo "  nats --server localhost:4222 account info"
    echo ""
else
    echo "💡 安装NATS CLI工具获取更多管理功能:"
    echo "  curl -sf https://binaries.nats.dev/nats-io/nats/v2@latest | sh"
    echo ""
fi

echo "✨ NATS集群部署完成！"