#!/usr/bin/env python3
"""
独立的故障容错API服务器，用于测试功能
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import uvicorn

# 简化的故障容错系统模拟
class SimpleFaultToleranceAPI:
    """简化的故障容错API实现"""
    
    def __init__(self):
        # 模拟数据
        self.system_status = {
            "status": "healthy",
            "system_started": True,
            "active_faults": 0,
            "total_components": 5,
            "healthy_components": 5,
            "last_updated": datetime.now().isoformat()
        }
        
        self.recovery_stats = {
            "total_recoveries": 10,
            "success_rate": 0.95,
            "avg_recovery_time": 45.2,
            "last_recovery": "2025-08-27T10:30:00Z"
        }
        
        self.backup_stats = {
            "successful_backups": 25,
            "failed_backups": 1,
            "total_size_gb": 128.5,
            "last_backup": "2025-08-27T11:00:00Z"
        }
        
        self.consistency_stats = {
            "consistency_rate": 0.98,
            "total_checks": 100,
            "inconsistent_checks": 2,
            "last_check": "2025-08-27T11:45:00Z"
        }
        
        self.fault_events = [
            {
                "id": "fault_001",
                "type": "network_timeout",
                "component": "service_discovery",
                "severity": "medium",
                "timestamp": "2025-08-27T10:15:00Z",
                "status": "resolved"
            },
            {
                "id": "fault_002", 
                "type": "memory_leak",
                "component": "task_coordinator",
                "severity": "low",
                "timestamp": "2025-08-27T09:30:00Z",
                "status": "resolved"
            }
        ]

def create_app():
    """创建FastAPI应用"""
    app = FastAPI(
        title="Fault Tolerance API",
        description="故障容错系统API",
        version="0.1.0"
    )
    
    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 创建故障容错API实例
    fault_api = SimpleFaultToleranceAPI()
    
    # 基础路由
    @app.get("/")
    async def root():
        return {"message": "Fault Tolerance API", "status": "running"}
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "fault-tolerance-api"}
    
    # 故障容错API路由
    @app.get("/api/v1/fault-tolerance/status")
    async def get_system_status():
        """获取系统状态"""
        return fault_api.system_status
    
    @app.get("/api/v1/fault-tolerance/recovery/statistics")
    async def get_recovery_statistics():
        """获取恢复统计信息"""
        return fault_api.recovery_stats
    
    @app.get("/api/v1/fault-tolerance/backup/statistics")
    async def get_backup_statistics():
        """获取备份统计信息"""
        return fault_api.backup_stats
    
    @app.get("/api/v1/fault-tolerance/consistency/statistics")
    async def get_consistency_statistics():
        """获取一致性统计信息"""
        return fault_api.consistency_stats
    
    @app.get("/api/v1/fault-tolerance/faults/events")
    async def get_fault_events():
        """获取故障事件列表"""
        return {"events": fault_api.fault_events}
    
    @app.post("/api/v1/fault-tolerance/testing/inject-fault")
    async def inject_fault(fault_data: Dict[str, Any]):
        """注入故障测试"""
        fault_id = f"test_fault_{len(fault_api.fault_events) + 1:03d}"
        new_fault = {
            "id": fault_id,
            "type": fault_data.get("fault_type", "test_fault"),
            "component": fault_data.get("component_id", "test_component"),
            "severity": fault_data.get("severity", "low"),
            "timestamp": datetime.now().isoformat(),
            "status": "injected"
        }
        fault_api.fault_events.append(new_fault)
        return {
            "success": True,
            "fault_id": fault_id,
            "message": "故障注入成功"
        }
    
    return app

if __name__ == "__main__":
    app = create_app()
    print("故障容错API服务器启动中...")
    print("API地址: http://localhost:8001")
    print("文档地址: http://localhost:8001/docs")
    print("测试路由:")
    print("  GET  /api/v1/fault-tolerance/status")
    print("  GET  /api/v1/fault-tolerance/recovery/statistics")
    print("  GET  /api/v1/fault-tolerance/backup/statistics")
    print("  GET  /api/v1/fault-tolerance/consistency/statistics")
    print("  POST /api/v1/fault-tolerance/testing/inject-fault")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )