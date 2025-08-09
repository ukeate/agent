#!/bin/bash

# 统一开发环境启动脚本
# 用法: ./scripts/dev.sh [mode]
# mode: all(默认) | web | api | shared | docker

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# 检查端口是否被占用
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null; then
        return 0
    else
        return 1
    fi
}

# 等待服务就绪
wait_for_service() {
    local url=$1
    local timeout=${2:-30}
    local count=0
    
    log "等待服务就绪: $url"
    
    while [ $count -lt $timeout ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            success "服务就绪: $url"
            return 0
        fi
        
        sleep 1
        count=$((count + 1))
        echo -n "."
    done
    
    echo ""
    warn "服务启动超时: $url"
    return 1
}

# 检查依赖
check_dependencies() {
    log "检查开发依赖..."
    
    if ! command -v node &> /dev/null; then
        error "Node.js 未安装"
        exit 1
    fi
    
    if ! command -v npm &> /dev/null; then
        error "npm 未安装"
        exit 1
    fi
    
    if [[ "$MODE" == "api" || "$MODE" == "all" ]]; then
        if ! command -v uv &> /dev/null; then
            error "uv 未安装，请先安装: curl -LsSf https://astral.sh/uv/install.sh | sh"
            exit 1
        fi
    fi
    
    if [[ "$MODE" == "docker" || "$MODE" == "all" ]]; then
        if ! command -v docker &> /dev/null; then
            warn "Docker 未安装，将跳过Docker服务"
        fi
    fi
    
    success "依赖检查完成"
}

# 安装依赖
install_dependencies() {
    log "安装项目依赖..."
    
    # 根级依赖
    if [ ! -d "node_modules" ]; then
        log "安装根级依赖..."
        npm install
    fi
    
    # 共享包依赖
    if [[ "$MODE" == "shared" || "$MODE" == "web" || "$MODE" == "all" ]]; then
        if [ -d "packages/shared" ] && [ ! -d "packages/shared/node_modules" ]; then
            log "安装共享包依赖..."
            cd packages/shared && npm install && cd -
        fi
    fi
    
    # 前端依赖
    if [[ "$MODE" == "web" || "$MODE" == "all" ]]; then
        if [ -d "apps/web" ] && [ ! -d "apps/web/node_modules" ]; then
            log "安装前端依赖..."
            cd apps/web && npm install && cd -
        fi
    fi
    
    # 后端依赖
    if [[ "$MODE" == "api" || "$MODE" == "all" ]]; then
        if [ -d "apps/api" ] && command -v uv &> /dev/null; then
            log "安装后端依赖..."
            cd apps/api && uv sync && cd -
        fi
    fi
    
    success "依赖安装完成"
}

# 启动Docker服务
start_docker_services() {
    if ! command -v docker &> /dev/null; then
        warn "Docker 未安装，跳过Docker服务"
        return 0
    fi
    
    log "启动Docker基础服务..."
    
    if [ -f "infrastructure/docker/docker-compose.yml" ]; then
        docker compose -f infrastructure/docker/docker-compose.yml up -d postgres redis qdrant
        
        # 等待数据库就绪
        log "等待数据库服务就绪..."
        sleep 5
        
        success "Docker服务启动完成"
    else
        warn "Docker配置文件不存在，跳过Docker服务"
    fi
}

# 启动共享包开发模式
start_shared_dev() {
    log "启动共享包开发模式..."
    
    if [ ! -d "packages/shared" ]; then
        warn "共享包目录不存在，跳过"
        return 0
    fi
    
    cd packages/shared
    
    if [ -f "package.json" ] && grep -q '"dev"' package.json; then
        npm run dev &
        SHARED_PID=$!
        echo $SHARED_PID > ../../.shared.pid
        success "共享包开发模式已启动 (PID: $SHARED_PID)"
    else
        warn "共享包没有dev脚本，跳过"
    fi
    
    cd - > /dev/null
}

# 启动前端开发服务器
start_web_dev() {
    log "启动前端开发服务器..."
    
    if [ ! -d "apps/web" ]; then
        error "前端目录不存在"
        return 1
    fi
    
    cd apps/web
    
    # 检查端口
    if check_port 3002; then
        warn "端口 3002 已被占用"
    fi
    
    npm run dev &
    WEB_PID=$!
    echo $WEB_PID > ../../.web.pid
    
    cd - > /dev/null
    
    success "前端开发服务器已启动 (PID: $WEB_PID)"
    log "前端访问地址: http://localhost:3002"
}

# 启动后端开发服务器
start_api_dev() {
    log "启动后端开发服务器..."
    
    if [ ! -d "apps/api" ]; then
        error "后端目录不存在"
        return 1
    fi
    
    cd apps/api/src
    
    # 检查端口
    if check_port 8000; then
        warn "端口 8000 已被占用"
    fi
    
    # 启动FastAPI服务器
    uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload > ../../../.api.log 2>&1 &
    API_PID=$!
    echo $API_PID > ../../../.api.pid
    
    cd - > /dev/null
    
    success "后端开发服务器已启动 (PID: $API_PID)"
    log "后端访问地址: http://localhost:8000"
    log "API文档地址: http://localhost:8000/docs"
}

# 显示服务状态
show_status() {
    log "服务状态检查..."
    
    # 检查前端
    if check_port 3002; then
        success "前端服务运行中 (http://localhost:3002)"
    else
        warn "前端服务未运行"
    fi
    
    # 检查后端
    if check_port 8000; then
        success "后端服务运行中 (http://localhost:8000)"
    else
        warn "后端服务未运行"
    fi
    
    # 检查数据库
    if check_port 5432; then
        success "PostgreSQL 运行中"
    else
        warn "PostgreSQL 未运行"
    fi
    
    # 检查Redis
    if check_port 6379; then
        success "Redis 运行中"
    else
        warn "Redis 未运行"
    fi
    
    # 检查Qdrant
    if check_port 6333; then
        success "Qdrant 运行中"
    else
        warn "Qdrant 未运行"
    fi
}

# 清理函数
cleanup() {
    log "正在停止开发服务器..."
    
    if [ -f ".web.pid" ]; then
        WEB_PID=$(cat .web.pid)
        kill $WEB_PID 2>/dev/null || true
        rm .web.pid
    fi
    
    if [ -f ".api.pid" ]; then
        API_PID=$(cat .api.pid)
        kill $API_PID 2>/dev/null || true
        rm .api.pid
    fi
    
    if [ -f ".shared.pid" ]; then
        SHARED_PID=$(cat .shared.pid)
        kill $SHARED_PID 2>/dev/null || true
        rm .shared.pid
    fi
    
    success "开发服务器已停止"
}

# 信号处理
trap cleanup EXIT INT TERM

# 显示帮助
show_help() {
    echo "AI Agent System 开发环境启动脚本"
    echo ""
    echo "用法: $0 [mode] [options]"
    echo ""
    echo "模式:"
    echo "  all     启动所有服务 (默认)"
    echo "  web     仅启动前端"
    echo "  api     仅启动后端"
    echo "  shared  仅启动共享包"
    echo "  docker  仅启动Docker服务"
    echo "  status  显示服务状态"
    echo ""
    echo "选项:"
    echo "  -h, --help    显示此帮助信息"
    echo "  --no-docker   不启动Docker服务"
    echo ""
    echo "示例:"
    echo "  $0              # 启动所有服务"
    echo "  $0 web          # 仅启动前端"
    echo "  $0 all --no-docker  # 启动所有服务但不启动Docker"
}

# 主函数
main() {
    MODE=${1:-all}
    NO_DOCKER=false
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            --no-docker)
                NO_DOCKER=true
                shift
                ;;
            status)
                show_status
                exit 0
                ;;
            *)
                if [[ -z "$MODE" ]]; then
                    MODE=$1
                fi
                shift
                ;;
        esac
    done
    
    log "启动 AI Agent System 开发环境"
    log "模式: $MODE"
    
    # 检查依赖
    check_dependencies
    
    # 安装依赖
    install_dependencies
    
    # 启动Docker服务（如果需要）
    if [[ "$NO_DOCKER" == "false" && ("$MODE" == "all" || "$MODE" == "docker") ]]; then
        start_docker_services
    fi
    
    # 根据模式启动服务
    case "$MODE" in
        "shared")
            start_shared_dev
            ;;
        "web")
            if [[ "$MODE" == "web" ]]; then
                start_shared_dev
            fi
            start_web_dev
            ;;
        "api")
            start_api_dev
            ;;
        "docker")
            # Docker服务已在上面启动
            success "Docker服务启动完成"
            ;;
        "all")
            start_shared_dev
            start_api_dev
            sleep 3
            start_web_dev
            ;;
        *)
            error "未知的模式: $MODE"
            show_help
            exit 1
            ;;
    esac
    
    # 等待服务就绪
    if [[ "$MODE" == "all" || "$MODE" == "api" ]]; then
        wait_for_service "http://localhost:8000/api/v1/multi-agent/health" 30
    fi
    
    if [[ "$MODE" == "all" || "$MODE" == "web" ]]; then
        wait_for_service "http://localhost:3002" 30
    fi
    
    # 显示最终状态
    show_status
    
    success "开发环境启动完成!"
    log "按 Ctrl+C 停止所有服务"
    
    # 保持脚本运行
    if [[ "$MODE" != "docker" ]]; then
        wait
    fi
}

# 运行主函数
main "$@"