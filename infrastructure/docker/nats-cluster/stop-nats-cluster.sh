#!/bin/bash

# NATS集群停止脚本
set -e

# 配置变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.nats.yml"

echo "🛑 停止NATS JetStream集群..."

# 优雅停止集群
echo "📤 正在优雅停止NATS节点..."
docker-compose -f "$COMPOSE_FILE" stop

echo "🧹 清理容器和网络..."
docker-compose -f "$COMPOSE_FILE" down --volumes --remove-orphans

# 可选：清理数据（谨慎使用）
read -p "是否清理JetStream数据? 这将删除所有消息数据 (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  清理JetStream数据..."
    rm -rf "${SCRIPT_DIR}/data/nats-*/jetstream/*"
    rm -rf "${SCRIPT_DIR}/data/nats-*/logs/*"
    echo "✅ 数据清理完成"
else
    echo "📦 保留JetStream数据"
fi

# 清理Docker资源
echo "🧼 清理未使用的Docker资源..."
docker container prune -f 2>/dev/null || true
docker network prune -f 2>/dev/null || true
docker volume prune -f 2>/dev/null || true

echo ""
echo "✅ NATS集群已停止"
echo ""
echo "💡 重新启动集群:"
echo "  ./start-nats-cluster.sh"
echo ""