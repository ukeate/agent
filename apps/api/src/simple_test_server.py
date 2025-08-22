#!/usr/bin/env python3
"""
简化的测试服务器，专门用于测试反馈系统API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 导入反馈系统路由
from src.api.v1.feedback import router as feedback_router

# 创建FastAPI应用
app = FastAPI(title="反馈系统测试服务器", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含反馈系统路由
app.include_router(feedback_router, prefix="/api/v1/feedback", tags=["feedback"])

@app.get("/")
async def root():
    return {"message": "反馈系统测试服务器运行中"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "feedback-test-server"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)