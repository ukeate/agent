#!/usr/bin/env python3
"""
简单的API服务器 - 用于测试前端功能
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from src.core.security.middleware import SecureHeadersMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import random
from src.core.config import get_settings
from src.core.utils.timezone_utils import utc_now

app = FastAPI(title="AI Agent Simple API", version="1.0.0")
settings = get_settings()

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
    expose_headers=settings.CORS_EXPOSE_HEADERS,
)
if settings.FORCE_HTTPS:
    app.add_middleware(HTTPSRedirectMiddleware)

if settings.TRUSTED_HOSTS:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.TRUSTED_HOSTS,
        www_redirect=settings.TRUSTED_HOSTS_WWW_REDIRECT,
    )

app.add_middleware(
    GZipMiddleware,
    minimum_size=settings.GZIP_MINIMUM_SIZE,
    compresslevel=settings.GZIP_COMPRESS_LEVEL,
)
app.add_middleware(SecureHeadersMiddleware)

# 存储对话状态
conversations = {}
agents_store = {}

# 数据模型
class CreateConversationRequest(BaseModel):
    message: str
    agent_roles: Optional[List[str]] = None
    max_rounds: Optional[int] = 10

class AgentRequest(BaseModel):
    name: str
    type: str
    config: Optional[Dict[str, Any]] = {}

# 基础端点
@app.get("/")
async def root():
    return {
        "message": "AI Agent Simple API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": utc_now().isoformat()
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "ai-agent-api",
        "version": "1.0.0",
        "services": {
            "database": "healthy",
            "redis": "healthy"
        },
        "timestamp": utc_now().isoformat()
    }

# 多智能体API
@app.post("/api/v1/multi-agent/conversation")
async def create_conversation(request: CreateConversationRequest):
    conversation_id = str(uuid.uuid4())

    # 创建模拟对话
    conversation = {
        "conversation_id": conversation_id,
        "status": "active",
        "created_at": utc_now().isoformat(),
        "message": request.message,
        "agent_roles": request.agent_roles or ["assistant", "critic"],
        "max_rounds": request.max_rounds,
        "round_count": 0,
        "messages": [
            {
                "role": "user",
                "content": request.message,
                "timestamp": utc_now().isoformat()
            }
        ],
        "participants": [
            {"name": "Assistant", "role": "assistant", "status": "active"},
            {"name": "Critic", "role": "critic", "status": "active"}
        ]
    }

    conversations[conversation_id] = conversation

    return {
        "conversation_id": conversation_id,
        "status": "active",
        "participants": conversation["participants"],
        "created_at": conversation["created_at"],
        "config": {
            "max_rounds": request.max_rounds,
            "auto_reply": True
        },
        "initial_status": {
            "message": "对话创建成功",
            "ready": True
        }
    }

@app.get("/api/v1/multi-agent/conversation/{conversation_id}/status")
async def get_conversation_status(conversation_id: str):
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="对话不存在")

    conv = conversations[conversation_id]

    # 模拟对话进展
    if conv["round_count"] < conv["max_rounds"]:
        conv["round_count"] += 1
        conv["messages"].append({
            "role": "assistant",
            "content": f"这是智能体的第{conv['round_count']}轮回复",
            "timestamp": utc_now().isoformat()
        })

    return {
        "conversation_id": conversation_id,
        "status": conv["status"],
        "created_at": conv["created_at"],
        "updated_at": utc_now().isoformat(),
        "message_count": len(conv["messages"]),
        "round_count": conv["round_count"],
        "participants": conv["participants"],
        "config": {
            "max_rounds": conv["max_rounds"],
            "auto_reply": True
        }
    }

@app.get("/api/v1/multi-agent/conversation/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str):
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="对话不存在")

    return {
        "conversation_id": conversation_id,
        "messages": conversations[conversation_id]["messages"],
        "total": len(conversations[conversation_id]["messages"])
    }

# 实验API
@app.get("/api/v1/experiments")
async def list_experiments():
    return {
        "experiments": [
            {
                "id": "exp-1",
                "name": "按钮颜色测试",
                "status": "active",
                "created_at": "2025-01-01T10:00:00Z",
                "variants": ["control", "blue", "green"],
                "traffic_allocation": [33.3, 33.3, 33.4],
                "metrics": {
                    "conversion_rate": 12.5,
                    "engagement": 45.2,
                    "revenue": 1500
                },
                "sample_size": 10000,
                "significance_level": 0.95
            },
            {
                "id": "exp-2",
                "name": "推荐算法对比",
                "status": "completed",
                "created_at": "2025-01-02T10:00:00Z",
                "variants": ["algorithm_a", "algorithm_b"],
                "traffic_allocation": [50, 50],
                "metrics": {
                    "conversion_rate": 15.3,
                    "engagement": 62.1,
                    "revenue": 2500
                },
                "sample_size": 5000,
                "significance_level": 0.99
            }
        ],
        "total": 2
    }

@app.post("/api/v1/experiments")
async def create_experiment(data: Dict[str, Any]):
    exp_id = f"exp-{str(uuid.uuid4())[:8]}"
    return {
        "id": exp_id,
        "name": data.get("name", "新实验"),
        "status": "draft",
        "created_at": utc_now().isoformat(),
        "config": data
    }

# RAG API
@app.post("/api/v1/rag/query")
async def rag_query(data: Dict[str, Any]):
    query = data.get("query", "")
    return {
        "query": query,
        "response": f"基于知识库的回答：{query}",
        "sources": [
            {"title": "文档1", "relevance": 0.95},
            {"title": "文档2", "relevance": 0.87}
        ],
        "confidence": 0.89
    }

@app.post("/api/v1/rag/documents")
async def upload_document(data: Dict[str, Any]):
    doc_id = str(uuid.uuid4())
    return {
        "document_id": doc_id,
        "status": "indexed",
        "chunks": random.randint(10, 100),
        "message": "文档上传成功"
    }

# 工作流API
@app.get("/api/v1/workflows")
async def list_workflows():
    return {
        "workflows": [
            {
                "id": "wf-1",
                "name": "数据处理流程",
                "status": "active",
                "steps": 5
            },
            {
                "id": "wf-2",
                "name": "模型训练流程",
                "status": "completed",
                "steps": 8
            }
        ],
        "total": 2
    }

@app.post("/api/v1/workflows/{workflow_id}/start")
async def start_workflow(workflow_id: str):
    return {
        "workflow_id": workflow_id,
        "execution_id": str(uuid.uuid4()),
        "status": "running",
        "started_at": utc_now().isoformat()
    }

# 智能体管理API
@app.post("/api/v1/agents")
async def create_agent(request: AgentRequest):
    agent_id = str(uuid.uuid4())
    agent = {
        "id": agent_id,
        "name": request.name,
        "type": request.type,
        "status": "active",
        "config": request.config,
        "created_at": utc_now().isoformat()
    }
    agents_store[agent_id] = agent
    return agent

@app.get("/api/v1/agents")
async def list_agents():
    return {
        "agents": list(agents_store.values()),
        "total": len(agents_store)
    }

# 监控API
@app.get("/api/v1/monitoring/metrics")
async def get_metrics():
    return {
        "metrics": {
            "cpu_usage": random.uniform(20, 80),
            "memory_usage": random.uniform(30, 70),
            "request_rate": random.randint(100, 1000),
            "error_rate": random.uniform(0, 5)
        },
        "timestamp": utc_now().isoformat()
    }

# 测试端点
@app.get("/api/v1/test/health")
async def test_health():
    return {"status": "ok", "timestamp": utc_now().isoformat()}

# 健康检查和监控端点
@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "ok",
        "database": {"status": "ok", "latency": random.randint(1, 20)},
        "redis": {"status": "ok", "latency": random.randint(1, 5)},
        "timestamp": utc_now().isoformat()
    }

@app.get("/api/v1/metrics")
async def get_system_metrics():
    return {
        "cpu": {"usage": random.uniform(30, 70)},
        "memory": {"usage": random.uniform(40, 80)},
        "disk": {"usage": random.uniform(20, 60)},
        "network": {"throughput": random.uniform(50, 200)},
        "connections": random.randint(100, 500),
        "requests": {"rate": random.randint(500, 2000)},
        "timestamp": utc_now().isoformat()
    }

@app.get("/api/v1/alerts")
async def get_alerts():
    return {
        "alerts": [
            {
                "id": "alert-1",
                "level": "warning",
                "message": "CPU使用率偏高",
                "service": "API Gateway",
                "timestamp": utc_now().isoformat(),
                "resolved": False
            },
            {
                "id": "alert-2",
                "level": "info",
                "message": "系统备份完成",
                "service": "Backup Service",
                "timestamp": utc_now().isoformat(),
                "resolved": True
            }
        ],
        "total": 2
    }

# 实验平台额外端点
@app.post("/api/v1/experiments/{experiment_id}/start")
async def start_experiment(experiment_id: str):
    return {
        "experiment_id": experiment_id,
        "status": "running",
        "started_at": utc_now().isoformat()
    }

@app.post("/api/v1/experiments/{experiment_id}/pause")
async def pause_experiment(experiment_id: str):
    return {
        "experiment_id": experiment_id,
        "status": "paused",
        "paused_at": utc_now().isoformat()
    }

@app.post("/api/v1/experiments/{experiment_id}/resume")
async def resume_experiment(experiment_id: str):
    return {
        "experiment_id": experiment_id,
        "status": "running",
        "resumed_at": utc_now().isoformat()
    }

# 工作流额外端点
@app.get("/api/v1/workflows/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    return {
        "workflow_id": workflow_id,
        "status": random.choice(["running", "completed", "failed"]),
        "progress": random.randint(0, 100),
        "current_step": random.randint(1, 5),
        "total_steps": 5
    }

@app.post("/api/v1/workflows")
async def create_workflow(data: Dict[str, Any]):
    workflow_id = f"wf-{str(uuid.uuid4())[:8]}"
    return {
        "workflow_id": workflow_id,
        "name": data.get("name", "新工作流"),
        "status": "draft",
        "created_at": utc_now().isoformat()
    }

# 多智能体协作额外端点
@app.post("/api/v1/multi-agent/conversation/{conversation_id}/pause")
async def pause_conversation(conversation_id: str):
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="对话不存在")

    conversations[conversation_id]["status"] = "paused"
    return {"status": "paused", "conversation_id": conversation_id}

@app.post("/api/v1/multi-agent/conversation/{conversation_id}/resume")
async def resume_conversation(conversation_id: str):
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="对话不存在")

    conversations[conversation_id]["status"] = "active"
    return {"status": "active", "conversation_id": conversation_id}

@app.post("/api/v1/multi-agent/conversation/{conversation_id}/terminate")
async def terminate_conversation(conversation_id: str):
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="对话不存在")

    conversations[conversation_id]["status"] = "terminated"
    return {"status": "terminated", "conversation_id": conversation_id}

# RAG额外端点
@app.delete("/api/v1/rag/documents/{document_id}")
async def delete_document(document_id: str):
    return {
        "document_id": document_id,
        "status": "deleted",
        "message": "文档删除成功"
    }

@app.get("/api/v1/rag/documents")
async def list_documents():
    return {
        "documents": [
            {
                "id": f"doc-{i}",
                "name": f"文档{i}.pdf",
                "size": random.randint(100000, 10000000),
                "chunks": random.randint(10, 100),
                "status": "indexed",
                "uploadedAt": utc_now().isoformat()
            }
            for i in range(1, 4)
        ],
        "total": 3
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
