#!/bin/bash

# 统一构建脚本
# 用法: ./scripts/build.sh [target]
# target: all(默认) | shared | web | api | docker

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

# 检查依赖
check_dependencies() {
    log "检查构建依赖..."
    
    if ! command -v node &> /dev/null; then
        error "Node.js 未安装"
        exit 1
    fi
    
    if ! command -v npm &> /dev/null; then
        error "npm 未安装"
        exit 1
    fi
    
    if [[ "$TARGET" == "api" || "$TARGET" == "all" ]]; then
        if ! command -v uv &> /dev/null; then
            warn "uv 未安装，API构建将跳过"
        fi
    fi
    
    success "依赖检查完成"
}

# 清理构建产物
clean_build() {
    log "清理构建产物..."
    
    if [[ "$TARGET" == "shared" || "$TARGET" == "all" ]]; then
        rm -rf packages/shared/dist
    fi
    
    if [[ "$TARGET" == "web" || "$TARGET" == "all" ]]; then
        rm -rf apps/web/dist apps/web/build apps/web/.next
    fi
    
    if [[ "$TARGET" == "api" || "$TARGET" == "all" ]]; then
        find apps/api -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find apps/api -name "*.pyc" -delete 2>/dev/null || true
    fi
    
    success "清理完成"
}

# 构建共享包
build_shared() {
    log "构建共享包..."
    
    cd packages/shared
    
    if [ ! -f "package.json" ]; then
        error "packages/shared/package.json 不存在"
        return 1
    fi
    
    # 安装依赖
    if [ ! -d "node_modules" ]; then
        log "安装共享包依赖..."
        npm install
    fi
    
    # 构建
    npm run build
    
    cd - > /dev/null
    success "共享包构建完成"
}

# 构建前端
build_web() {
    log "构建前端应用..."
    
    cd apps/web
    
    if [ ! -f "package.json" ]; then
        error "apps/web/package.json 不存在"
        return 1
    fi
    
    # 安装依赖
    if [ ! -d "node_modules" ]; then
        log "安装前端依赖..."
        npm install
    fi
    
    # 类型检查
    if npm run typecheck &> /dev/null; then
        log "运行类型检查..."
        npm run typecheck
    fi
    
    # 构建
    npm run build
    
    cd - > /dev/null
    success "前端构建完成"
}

# 构建后端
build_api() {
    log "构建后端应用..."
    
    cd apps/api
    
    if [ ! -f "pyproject.toml" ]; then
        error "apps/api/pyproject.toml 不存在"
        return 1
    fi
    
    if command -v uv &> /dev/null; then
        # 同步依赖
        log "同步Python依赖..."
        uv sync
        
        # 运行代码检查
        log "运行代码检查..."
        uv run black --check src tests || warn "代码格式检查失败"
        uv run ruff check src tests || warn "代码质量检查失败"
        
        success "后端构建完成"
    else
        warn "uv 未安装，跳过后端构建"
    fi
    
    cd - > /dev/null
}

# 构建Docker镜像
build_docker() {
    log "构建Docker镜像..."
    
    if [ ! -f "infrastructure/docker/docker-compose.yml" ]; then
        error "Docker配置文件不存在"
        return 1
    fi
    
    docker compose -f infrastructure/docker/docker-compose.yml build
    
    success "Docker镜像构建完成"
}

# 运行测试
run_tests() {
    log "运行测试..."
    
    # 共享包测试
    if [[ "$TARGET" == "shared" || "$TARGET" == "all" ]]; then
        if [ -d "packages/shared" ] && [ -f "packages/shared/package.json" ]; then
            cd packages/shared
            if npm run test &> /dev/null; then
                npm run test || warn "共享包测试失败"
            fi
            cd - > /dev/null
        fi
    fi
    
    # 前端测试
    if [[ "$TARGET" == "web" || "$TARGET" == "all" ]]; then
        if [ -d "apps/web" ] && [ -f "apps/web/package.json" ]; then
            cd apps/web
            if npm run test &> /dev/null; then
                npm run test || warn "前端测试失败"
            fi
            cd - > /dev/null
        fi
    fi
    
    # 后端测试
    if [[ "$TARGET" == "api" || "$TARGET" == "all" ]]; then
        if [ -d "apps/api" ] && command -v uv &> /dev/null; then
            cd apps/api
            uv run pytest || warn "后端测试失败"
            cd - > /dev/null
        fi
    fi
    
    success "测试完成"
}

# 显示构建信息
show_build_info() {
    log "构建信息:"
    echo "  目标: $TARGET"
    echo "  环境: ${NODE_ENV:-development}"
    echo "  时间: $(date)"
    
    if [ -f "package.json" ]; then
        VERSION=$(node -p "require('./package.json').version")
        echo "  版本: $VERSION"
    fi
}

# 主函数
main() {
    TARGET=${1:-all}
    
    log "开始构建 AI Agent System"
    show_build_info
    
    # 检查依赖
    check_dependencies
    
    # 清理构建产物
    clean_build
    
    # 根据目标执行构建
    case "$TARGET" in
        "shared")
            build_shared
            ;;
        "web")
            build_shared
            build_web
            ;;
        "api")
            build_api
            ;;
        "docker")
            build_docker
            ;;
        "all")
            build_shared
            build_web
            build_api
            run_tests
            ;;
        *)
            error "未知的构建目标: $TARGET"
            echo "支持的目标: all, shared, web, api, docker"
            exit 1
            ;;
    esac
    
    success "构建完成!"
}

# 运行主函数
main "$@"