#!/usr/bin/env python3
"""
简化的API服务器用于测试故障容错功能
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

def create_test_app():
    """创建测试应用"""
    app = FastAPI(title="Fault Tolerance Test API", version="0.1.0")
    
    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 初始化故障容错系统
    from core.dependencies import initialize_fault_tolerance_system
    initialize_fault_tolerance_system()
    
    # 注册故障容错路由
    from api.v1.fault_tolerance import router
    app.include_router(router, prefix="/api/v1")
    
    @app.get("/")
    async def root():
        return {"message": "Fault Tolerance Test API", "status": "running"}
    
    @app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    return app

if __name__ == "__main__":
    app = create_test_app()
    print("故障容错测试API服务器启动中...")
    print("API地址: http://localhost:8001")
    print("文档地址: http://localhost:8001/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )