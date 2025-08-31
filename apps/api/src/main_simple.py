#!/usr/bin/env python3
"""
简化版本的API服务器，包含分布式任务协调引擎
"""

import sys
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import random

app = FastAPI(title="AI Agent System API", version="1.0.0")

# 添加CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 内存存储
tasks_store: Dict[str, Dict] = {}
agents_store: Dict[str, Dict] = {}
raft_nodes: Dict[str, Dict] = {}
system_stats = {
    "total_tasks": 0,
    "running_tasks": 0,
    "completed_tasks": 0,
    "failed_tasks": 0,
    "cluster_health": "healthy",
    "leader_node": "node_1",
    "active_nodes": 3,
    "total_nodes": 3
}

# 数据模型
class TaskRequest(BaseModel):
    task_type: str
    data: Dict[str, Any]
    requirements: Dict[str, Any] = {}
    priority: str = "medium"

class TaskResponse(BaseModel):
    task_id: str
    status: str
    progress: int = 0
    assigned_agent: Optional[str] = None
    created_at: datetime
    estimated_completion: Optional[datetime] = None

@app.get("/")
async def root():
    return {"message": "AI Agent System API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# 分布式任务协调引擎 API
@app.post("/api/v1/distributed-task/tasks", response_model=TaskResponse)
async def submit_task(task: TaskRequest, background_tasks: BackgroundTasks):
    """提交新任务"""
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    
    new_task = {
        "task_id": task_id,
        "task_type": task.task_type,
        "data": task.data,
        "requirements": task.requirements,
        "priority": task.priority,
        "status": "pending",
        "progress": 0,
        "assigned_agent": None,
        "created_at": datetime.now(),
        "estimated_completion": datetime.now() + timedelta(minutes=random.randint(5, 30))
    }
    
    tasks_store[task_id] = new_task
    system_stats["total_tasks"] += 1
    
    # 模拟后台任务处理
    background_tasks.add_task(simulate_task_processing, task_id)
    
    return TaskResponse(**new_task)

@app.get("/api/v1/distributed-task/tasks")
async def get_tasks(status: Optional[str] = None, limit: int = 100):
    """获取任务列表"""
    tasks = list(tasks_store.values())
    if status:
        tasks = [t for t in tasks if t["status"] == status]
    
    return {
        "success": True,
        "data": {
            "tasks": tasks[:limit],
            "total": len(tasks)
        }
    }

@app.get("/api/v1/distributed-task/tasks/{task_id}")
async def get_task(task_id: str):
    """获取单个任务详情"""
    if task_id not in tasks_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {"success": True, "data": tasks_store[task_id]}

@app.get("/api/v1/distributed-task/stats")
async def get_system_stats():
    """获取系统统计信息"""
    # 更新实时统计
    running_count = len([t for t in tasks_store.values() if t["status"] == "running"])
    completed_count = len([t for t in tasks_store.values() if t["status"] == "completed"])
    failed_count = len([t for t in tasks_store.values() if t["status"] == "failed"])
    
    system_stats.update({
        "running_tasks": running_count,
        "completed_tasks": completed_count,
        "failed_tasks": failed_count,
        "total_tasks": len(tasks_store),
        "avg_completion_time": random.randint(120, 300),
        "throughput": random.randint(50, 150),
        "error_rate": round(random.uniform(0.5, 2.0), 2)
    })
    
    return {"success": True, "data": system_stats}

# 任务分解器 API
@app.post("/api/v1/distributed-task/decomposer/decompose")
async def decompose_task(task_data: Dict[str, Any]):
    """任务分解"""
    task_type = task_data.get("task_type", "default")
    strategies = ["parallel", "sequential", "hierarchical", "pipeline"]
    
    # 模拟分解结果
    subtasks = []
    for i in range(random.randint(2, 5)):
        subtasks.append({
            "subtask_id": f"subtask_{uuid.uuid4().hex[:6]}",
            "task_type": f"{task_type}_step_{i+1}",
            "dependencies": [subtasks[-1]["subtask_id"]] if subtasks and random.choice([True, False]) else [],
            "estimated_duration": random.randint(30, 180),
            "priority": random.choice(["high", "medium", "low"]),
            "resource_requirements": {
                "cpu": random.randint(1, 4),
                "memory": random.randint(512, 2048),
                "gpu": random.choice([0, 1])
            }
        })
    
    return {
        "success": True,
        "data": {
            "original_task": task_data,
            "strategy": random.choice(strategies),
            "subtasks": subtasks,
            "execution_plan": {
                "estimated_total_time": sum(st["estimated_duration"] for st in subtasks),
                "parallelizable": random.choice([True, False]),
                "complexity_score": random.randint(1, 10)
            }
        }
    }

# 智能分配器 API
@app.get("/api/v1/distributed-task/assigner/agents")
async def get_available_agents():
    """获取可用智能体列表"""
    if not agents_store:
        # 初始化一些模拟智能体
        for i in range(8):
            agent_id = f"agent_{i+1}"
            agents_store[agent_id] = {
                "agent_id": agent_id,
                "status": random.choice(["idle", "busy", "offline"]),
                "capabilities": random.sample(["nlp", "vision", "reasoning", "planning", "execution"], k=random.randint(2, 4)),
                "current_load": random.randint(0, 100),
                "performance_score": round(random.uniform(0.7, 0.95), 2),
                "last_activity": datetime.now() - timedelta(minutes=random.randint(1, 60)),
                "completed_tasks": random.randint(50, 500),
                "success_rate": round(random.uniform(0.85, 0.98), 2)
            }
    
    return {"success": True, "data": list(agents_store.values())}

@app.post("/api/v1/distributed-task/assigner/assign")
async def assign_task(assignment_data: Dict[str, Any]):
    """分配任务给智能体"""
    task_id = assignment_data.get("task_id")
    strategy = assignment_data.get("strategy", "load_balanced")
    
    selected_agent = None
    if task_id and task_id in tasks_store:
        # 选择最佳智能体
        available_agents = [a for a in agents_store.values() if a["status"] == "idle"]
        if available_agents:
            selected_agent = min(available_agents, key=lambda x: x["current_load"])
            
            # 更新任务和智能体状态
            tasks_store[task_id]["assigned_agent"] = selected_agent["agent_id"]
            tasks_store[task_id]["status"] = "assigned"
            agents_store[selected_agent["agent_id"]]["status"] = "busy"
            agents_store[selected_agent["agent_id"]]["current_load"] += random.randint(20, 50)
    
    return {
        "success": True,
        "data": {
            "task_id": task_id,
            "assigned_agent": selected_agent["agent_id"] if selected_agent else None,
            "assignment_strategy": strategy,
            "assignment_time": datetime.now().isoformat(),
            "estimated_start_time": (datetime.now() + timedelta(minutes=random.randint(1, 5))).isoformat()
        }
    }

# Raft共识算法 API
@app.get("/api/v1/distributed-task/raft/cluster")
async def get_raft_cluster():
    """获取Raft集群状态"""
    if not raft_nodes:
        # 初始化Raft节点
        for i in range(3):
            node_id = f"node_{i+1}"
            raft_nodes[node_id] = {
                "node_id": node_id,
                "role": "leader" if i == 0 else "follower",
                "term": 5,
                "log_index": random.randint(100, 200),
                "commit_index": random.randint(90, 190),
                "last_heartbeat": datetime.now(),
                "status": "active",
                "vote_count": 3 if i == 0 else 0
            }
    
    return {
        "success": True,
        "data": {
            "cluster_status": "stable",
            "leader_node": "node_1",
            "current_term": 5,
            "nodes": list(raft_nodes.values()),
            "consensus_health": "healthy"
        }
    }

@app.post("/api/v1/distributed-task/raft/election")
async def trigger_leader_election():
    """触发领导者选举"""
    # 模拟选举过程
    new_leader = random.choice(list(raft_nodes.keys()))
    
    for node_id, node in raft_nodes.items():
        if node_id == new_leader:
            node["role"] = "leader"
            node["vote_count"] = len(raft_nodes)
            node["term"] += 1
        else:
            node["role"] = "follower"
            node["vote_count"] = 0
            node["term"] += 1
    
    return {
        "success": True,
        "data": {
            "new_leader": new_leader,
            "election_term": raft_nodes[new_leader]["term"],
            "election_time": datetime.now().isoformat()
        }
    }

# 状态管理器 API
@app.get("/api/v1/distributed-task/state/status")
async def get_distributed_state():
    """获取分布式状态"""
    return {
        "success": True,
        "data": {
            "global_state": {
                "active_tasks": len([t for t in tasks_store.values() if t["status"] in ["running", "assigned"]]),
                "pending_operations": random.randint(0, 10),
                "locked_resources": random.randint(0, 5),
                "checkpoint_status": "healthy"
            },
            "node_states": [
                {
                    "node_id": f"node_{i+1}",
                    "state_version": random.randint(100, 200),
                    "last_sync": datetime.now() - timedelta(seconds=random.randint(1, 30)),
                    "sync_status": "synchronized"
                } for i in range(3)
            ]
        }
    }

# 冲突解决器 API
@app.get("/api/v1/distributed-task/conflicts")
async def get_conflicts():
    """获取系统冲突列表"""
    conflicts = []
    for i in range(random.randint(0, 3)):
        conflicts.append({
            "conflict_id": f"conflict_{uuid.uuid4().hex[:6]}",
            "type": random.choice(["resource", "assignment", "state", "dependency"]),
            "severity": random.choice(["low", "medium", "high"]),
            "description": f"模拟冲突 #{i+1}",
            "involved_tasks": random.sample(list(tasks_store.keys()), k=min(2, len(tasks_store))),
            "detected_at": datetime.now() - timedelta(minutes=random.randint(1, 30)),
            "status": random.choice(["detected", "resolving", "resolved"])
        })
    
    return {"success": True, "data": conflicts}

@app.post("/api/v1/distributed-task/conflicts/resolve")
async def resolve_conflict(conflict_data: Dict[str, Any]):
    """解决冲突"""
    conflict_id = conflict_data.get("conflict_id")
    resolution_strategy = conflict_data.get("strategy", "priority_based")
    
    return {
        "success": True,
        "data": {
            "conflict_id": conflict_id,
            "resolution_strategy": resolution_strategy,
            "resolution_time": datetime.now().isoformat(),
            "result": "resolved",
            "actions_taken": [
                f"应用{resolution_strategy}策略",
                "重新分配资源",
                "更新任务优先级"
            ]
        }
    }

# 后台任务处理
async def simulate_task_processing(task_id: str):
    """模拟任务处理过程"""
    await asyncio.sleep(1)  # 初始延迟
    
    if task_id in tasks_store:
        task = tasks_store[task_id]
        
        # 分配阶段
        task["status"] = "assigned"
        await asyncio.sleep(2)
        
        # 运行阶段
        task["status"] = "running"
        system_stats["running_tasks"] += 1
        
        # 模拟进度更新
        for progress in [25, 50, 75, 100]:
            await asyncio.sleep(random.randint(2, 5))
            if task_id in tasks_store:
                task["progress"] = progress
        
        # 完成
        if task_id in tasks_store:
            task["status"] = "completed" if random.random() > 0.1 else "failed"
            task["progress"] = 100 if task["status"] == "completed" else task["progress"]
            system_stats["running_tasks"] = max(0, system_stats["running_tasks"] - 1)
            
            if task["status"] == "completed":
                system_stats["completed_tasks"] += 1
            else:
                system_stats["failed_tasks"] += 1

# 流式处理系统 API
@app.get("/api/v1/streaming/metrics")
async def get_streaming_metrics():
    """获取流式处理系统指标"""
    return {
        "system_metrics": {
            "total_sessions": 0,
            "active_sessions": random.randint(0, 5),
            "total_sessions_created": random.randint(50, 100),
            "total_tokens_processed": random.randint(10000, 50000),
            "total_events_processed": random.randint(500, 2000),
            "total_buffer_usage": random.randint(100, 500),
            "active_streamers": random.randint(0, 3),
            "active_buffers": random.randint(1, 5),
            "uptime": random.randint(3600, 86400)
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/streaming/backpressure/status")
async def get_backpressure_status():
    """获取背压控制状态"""
    return {
        "backpressure_enabled": False,
        "backpressure_status": {
            "throttle_level": "none",
            "buffer_usage": random.randint(20, 80),
            "buffer_usage_ratio": random.uniform(0.2, 0.8),
            "is_monitoring": True,
            "pressure_metrics": {
                "cpu_usage": {
                    "current_value": random.uniform(30, 80),
                    "threshold": 85.0,
                    "severity": random.randint(0, 3),
                    "over_threshold": False
                },
                "memory_usage": {
                    "current_value": random.uniform(40, 75),
                    "threshold": 90.0,
                    "severity": random.randint(0, 2),
                    "over_threshold": False
                }
            },
            "active_throttles": []
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/streaming/flow-control/metrics")
async def get_flow_control_metrics():
    """获取流量控制指标"""
    return {
        "flow_control_metrics": {
            "buffer_size": 0,
            "queue_depth": 0,
            "processing_rate": 0
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/streaming/queue/status")
async def get_queue_status():
    """获取队列状态"""
    queue_names = ["high_priority", "normal", "low_priority", "background"]
    queue_metrics = {}
    
    for queue_name in queue_names:
        queue_metrics[queue_name] = {
            "name": queue_name,
            "current_size": random.randint(0, 50),
            "max_size": random.randint(100, 200),
            "utilization": random.uniform(0.1, 0.8),
            "enqueue_rate": random.uniform(1.0, 10.0),
            "dequeue_rate": random.uniform(1.0, 12.0),
            "average_wait_time": random.uniform(0.5, 5.0),
            "oldest_item_age": random.uniform(0.1, 10.0),
            "is_overloaded": random.choice([True, False]),
            "throughput_ratio": random.uniform(0.8, 1.2)
        }
    
    overloaded_queues = [name for name, metrics in queue_metrics.items() if metrics["is_overloaded"]]
    
    return {
        "queue_metrics": queue_metrics,
        "system_summary": {
            "total_queues": len(queue_names),
            "overloaded_queues": len(overloaded_queues),
            "overloaded_queue_names": overloaded_queues,
            "total_items": sum(metrics["current_size"] for metrics in queue_metrics.values()),
            "total_capacity": sum(metrics["max_size"] for metrics in queue_metrics.values()),
            "system_utilization": sum(metrics["utilization"] for metrics in queue_metrics.values()) / len(queue_names),
            "average_utilization": sum(metrics["utilization"] for metrics in queue_metrics.values()) / len(queue_names),
            "is_running": True
        },
        "overloaded_queues": overloaded_queues,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/streaming/sessions")
async def get_streaming_sessions():
    """获取流式会话列表"""
    return {
        "sessions": {},
        "total_sessions": 0,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/streaming/sessions/{session_id}/metrics")
async def get_session_metrics(session_id: str):
    """获取指定会话指标"""
    return {
        "session_metrics": {
            "session_id": session_id,
            "active": random.choice([True, False]),
            "start_time": (datetime.now() - timedelta(minutes=random.randint(5, 60))).isoformat(),
            "messages_count": random.randint(0, 50)
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/streaming/health")
async def get_streaming_health():
    """获取流式处理系统健康状态"""
    return {
        "status": "healthy",
        "service": "streaming",
        "active_sessions": random.randint(0, 5),
        "total_sessions": 0,
        "uptime": random.randint(3600, 86400),
        "timestamp": datetime.now().isoformat()
    }

# 集群管理 API
@app.get("/api/v1/cluster/status")
async def get_cluster_status():
    """获取集群状态"""
    return {
        "success": True,
        "data": {
            "cluster_id": "ai-agent-cluster",
            "status": "healthy",
            "leader_node": "node_1",
            "total_nodes": 3,
            "active_nodes": 3,
            "consensus_state": "stable",
            "last_election": (datetime.now() - timedelta(hours=2)).isoformat(),
            "uptime": random.randint(86400, 604800),
            "version": "1.0.0"
        }
    }

@app.get("/api/v1/cluster/agents")
async def get_cluster_agents():
    """获取集群代理信息"""
    agents = []
    for i in range(3):
        agent_id = f"agent_{i+1}"
        agents.append({
            "agent_id": agent_id,
            "node_id": f"node_{i+1}",
            "status": random.choice(["active", "idle", "busy"]),
            "capabilities": ["nlp", "reasoning", "planning"],
            "current_load": random.randint(0, 100),
            "max_capacity": 100,
            "version": "1.0.0",
            "last_heartbeat": datetime.now().isoformat(),
            "uptime": random.randint(3600, 86400)
        })
    
    return {
        "success": True,
        "data": {
            "total_agents": len(agents),
            "active_agents": len([a for a in agents if a["status"] == "active"]),
            "agents": agents
        }
    }

@app.post("/api/v1/cluster/metrics/query")
async def query_cluster_metrics(request: Dict[str, Any] = {}):
    """查询集群指标"""
    metrics = {
        "cpu_usage": {
            "current": random.uniform(20, 80),
            "average": random.uniform(30, 70),
            "peak": random.uniform(60, 95)
        },
        "memory_usage": {
            "current": random.uniform(40, 85),
            "available": random.uniform(15, 60),
            "total": 16384  # MB
        },
        "network_io": {
            "bytes_sent": random.randint(1000000, 10000000),
            "bytes_received": random.randint(1000000, 10000000),
            "packets_sent": random.randint(10000, 100000),
            "packets_received": random.randint(10000, 100000)
        },
        "disk_usage": {
            "total": 1024000,  # MB
            "used": random.randint(100000, 800000),
            "available": random.randint(200000, 900000)
        },
        "task_metrics": {
            "total_processed": random.randint(1000, 5000),
            "success_rate": random.uniform(85, 99),
            "average_response_time": random.uniform(100, 2000),  # ms
            "throughput": random.uniform(10, 100)  # tasks/sec
        }
    }
    
    return {
        "success": True,
        "data": {
            "timestamp": datetime.now().isoformat(),
            "cluster_id": "ai-agent-cluster",
            "metrics": metrics,
            "query_params": request
        }
    }

# 多智能体 API
@app.get("/api/v1/multi-agent/agents")
async def get_multi_agent_agents():
    """获取多智能体列表"""
    agents = []
    for i in range(3):
        agent_id = f"agent_{i+1}"
        agents.append({
            "id": agent_id,
            "name": f"智能体 {i+1}",
            "type": random.choice(["assistant", "critic", "executor"]),
            "status": random.choice(["active", "idle", "busy"]),
            "capabilities": random.sample(["reasoning", "coding", "analysis", "planning"], random.randint(2, 4)),
            "current_task": f"task_{random.randint(1, 100)}" if random.random() > 0.3 else None,
            "performance_score": random.uniform(0.7, 0.98),
            "last_active": (datetime.now() - timedelta(minutes=random.randint(1, 60))).isoformat()
        })
    
    return {
        "success": True,
        "data": {
            "agents": agents,
            "total": len(agents),
            "active": len([a for a in agents if a["status"] == "active"])
        }
    }

# 监督者 API
@app.get("/api/v1/supervisor/status")
async def get_supervisor_status(supervisor_id: str = "main_supervisor"):
    """获取监督者状态"""
    return {
        "supervisor_id": supervisor_id,
        "status": "active",
        "uptime": random.randint(3600, 86400),
        "managed_agents": random.randint(3, 8),
        "active_tasks": random.randint(0, 5),
        "last_decision": (datetime.now() - timedelta(minutes=random.randint(1, 30))).isoformat(),
        "health": "good"
    }

@app.get("/api/v1/supervisor/stats")
async def get_supervisor_stats(supervisor_id: str = "main_supervisor"):
    """获取监督者统计信息"""
    return {
        "supervisor_id": supervisor_id,
        "total_tasks_assigned": random.randint(100, 500),
        "successful_completions": random.randint(80, 450),
        "failed_tasks": random.randint(5, 20),
        "average_completion_time": random.uniform(120, 600),
        "agent_utilization": random.uniform(0.6, 0.9),
        "success_rate": random.uniform(0.85, 0.98)
    }

@app.get("/api/v1/supervisor/metrics")
async def get_supervisor_metrics(supervisor_id: str = "main_supervisor"):
    """获取监督者指标"""
    return {
        "supervisor_id": supervisor_id,
        "current_load": random.uniform(0.3, 0.8),
        "decision_latency": random.uniform(50, 200),
        "throughput": random.uniform(10, 50),
        "error_rate": random.uniform(0.01, 0.05),
        "resource_usage": {
            "cpu": random.uniform(0.2, 0.7),
            "memory": random.uniform(0.4, 0.8)
        }
    }

@app.get("/api/v1/supervisor/tasks")
async def get_supervisor_tasks(supervisor_id: str = "main_supervisor", limit: int = 20, offset: int = 0):
    """获取监督者任务列表"""
    tasks = []
    for i in range(min(limit, random.randint(5, 15))):
        task_id = f"task_{i + offset + 1}"
        tasks.append({
            "task_id": task_id,
            "status": random.choice(["pending", "running", "completed", "failed"]),
            "assigned_agent": f"agent_{random.randint(1, 3)}",
            "priority": random.choice(["low", "medium", "high"]),
            "created_at": (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat(),
            "estimated_duration": random.randint(60, 1800),
            "progress": random.uniform(0, 1) if random.random() > 0.3 else 0
        })
    
    return {
        "supervisor_id": supervisor_id,
        "tasks": tasks,
        "total": random.randint(50, 200),
        "pagination": {"limit": limit, "offset": offset}
    }

@app.get("/api/v1/supervisor/decisions")
async def get_supervisor_decisions(supervisor_id: str = "main_supervisor", limit: int = 20, offset: int = 0):
    """获取监督者决策列表"""
    decisions = []
    for i in range(min(limit, random.randint(3, 10))):
        decision_id = f"decision_{i + offset + 1}"
        decisions.append({
            "decision_id": decision_id,
            "type": random.choice(["task_assignment", "resource_allocation", "conflict_resolution"]),
            "timestamp": (datetime.now() - timedelta(minutes=random.randint(1, 120))).isoformat(),
            "confidence": random.uniform(0.7, 0.99),
            "outcome": random.choice(["successful", "pending", "requires_review"]),
            "reasoning": f"基于当前系统状态和agent性能指标的决策 #{i+1}"
        })
    
    return {
        "supervisor_id": supervisor_id,
        "decisions": decisions,
        "total": random.randint(30, 150),
        "pagination": {"limit": limit, "offset": offset}
    }

@app.get("/api/v1/supervisor/config")
async def get_supervisor_config(supervisor_id: str = "main_supervisor"):
    """获取监督者配置"""
    return {
        "supervisor_id": supervisor_id,
        "config": {
            "max_concurrent_tasks": random.randint(5, 20),
            "decision_timeout": random.randint(30, 120),
            "retry_attempts": random.randint(2, 5),
            "agent_health_check_interval": random.randint(30, 300),
            "load_balancing_strategy": random.choice(["round_robin", "least_loaded", "capability_based"]),
            "escalation_threshold": random.uniform(0.8, 0.95)
        },
        "last_updated": (datetime.now() - timedelta(hours=random.randint(1, 48))).isoformat()
    }

# 事件系统 API
@app.get("/api/v1/events/list")
async def get_events_list(limit: int = 100):
    """获取事件列表"""
    events = []
    for i in range(min(limit, random.randint(5, 20))):
        event_id = f"event_{i+1}"
        events.append({
            "event_id": event_id,
            "type": random.choice(["MESSAGE_SENT", "TASK_COMPLETED", "AGENT_STATUS_CHANGED", "ERROR_OCCURRED"]),
            "source": random.choice(["agent_1", "agent_2", "supervisor", "system"]),
            "timestamp": (datetime.now() - timedelta(minutes=random.randint(1, 180))).isoformat(),
            "priority": random.choice(["low", "medium", "high", "critical"]),
            "message": f"事件消息 #{i+1}",
            "metadata": {
                "session_id": f"session_{random.randint(1, 50)}",
                "user_id": f"user_{random.randint(1, 20)}"
            }
        })
    
    return {
        "success": True,
        "data": {
            "events": events,
            "total": random.randint(100, 1000),
            "limit": limit
        }
    }

@app.get("/api/v1/events/stats")
async def get_events_stats(hours: int = 24):
    """获取事件统计信息"""
    return {
        "success": True,
        "data": {
            "time_window": f"{hours}h",
            "total_events": random.randint(100, 2000),
            "events_by_type": {
                "MESSAGE_SENT": random.randint(50, 800),
                "TASK_COMPLETED": random.randint(20, 400),
                "AGENT_STATUS_CHANGED": random.randint(10, 200),
                "ERROR_OCCURRED": random.randint(1, 50)
            },
            "events_by_priority": {
                "low": random.randint(50, 600),
                "medium": random.randint(30, 400),
                "high": random.randint(10, 200),
                "critical": random.randint(1, 20)
            },
            "error_rate": random.uniform(0.01, 0.05),
            "average_processing_time": random.uniform(10, 100)
        }
    }

@app.get("/api/v1/events/cluster/status")
async def get_events_cluster_status():
    """获取事件集群状态"""
    return {
        "success": True,
        "data": {
            "cluster_health": "healthy",
            "active_processors": random.randint(2, 5),
            "queue_depth": random.randint(0, 100),
            "processing_rate": random.uniform(50, 200),
            "last_processed": (datetime.now() - timedelta(seconds=random.randint(1, 30))).isoformat(),
            "uptime": random.randint(3600, 86400)
        }
    }

@app.post("/api/v1/events/submit")
async def submit_event(
    event_type: str,
    source: str,
    message: str,
    priority: str = "normal"
):
    """提交新事件"""
    event_id = f"event_{uuid.uuid4().hex[:8]}"
    
    return {
        "success": True,
        "data": {
            "event_id": event_id,
            "status": "accepted",
            "timestamp": datetime.now().isoformat(),
            "processing_queue_position": random.randint(1, 10)
        }
    }

# 智能体聊天 API
@app.post("/api/v1/agent/chat")
async def agent_chat(request: Dict[str, Any]):
    """智能体聊天接口"""
    message = request.get("message", "")
    agent_id = request.get("agent_id", "default_agent")
    session_id = request.get("session_id", f"session_{uuid.uuid4().hex[:8]}")
    
    # 模拟AI响应
    responses = [
        "我理解您的问题，让我来帮您分析一下。",
        "根据当前的情况，我建议采用以下方案：",
        "这是一个很有趣的问题，需要综合考虑多个因素。",
        "让我为您提供一个详细的解决方案。",
        "基于我的分析，这里有几个关键点需要注意：",
    ]
    
    return {
        "success": True,
        "data": {
            "session_id": session_id,
            "agent_id": agent_id,
            "response": random.choice(responses),
            "timestamp": datetime.now().isoformat(),
            "message_id": f"msg_{uuid.uuid4().hex[:8]}",
            "processing_time": random.uniform(0.5, 3.0)
        }
    }

# 故障容错系统 API
fault_tolerance_stats = {
    "recovery_stats": {
        "total_recoveries": 10,
        "success_rate": 0.95,
        "avg_recovery_time": 45.2,
        "last_recovery": "2025-08-27T10:30:00Z",
        "strategy_success_rates": {
            "immediate_restart": 0.98,
            "graceful_restart": 0.92,
            "task_migration": 0.87,
            "service_degradation": 0.78,
            "manual_intervention": 0.65
        },
        "recent_recoveries": []
    },
    "backup_stats": {
        "total_backups": 26,
        "total_size": 128.5,
        "success_rate": 0.96,
        "last_backup_time": "2025-08-27T11:00:00Z",
        "components": {
            "database": {
                "backup_count": 15,
                "last_backup": "2025-08-27T11:00:00Z",
                "total_size": 78.5
            },
            "vector_store": {
                "backup_count": 8,
                "last_backup": "2025-08-27T10:30:00Z",
                "total_size": 35.2
            },
            "config": {
                "backup_count": 3,
                "last_backup": "2025-08-27T10:00:00Z",
                "total_size": 14.8
            }
        }
    },
    "consistency_stats": {
        "total_checks": 100,
        "consistency_rate": 0.98,
        "last_check_time": "2025-08-27T11:45:00Z",
        "avg_check_duration": 2.5,
        "components_status": {
            "database": {
                "consistent": True,
                "last_check": "2025-08-27T11:45:00Z",
                "inconsistency_count": 0
            },
            "vector_store": {
                "consistent": True,
                "last_check": "2025-08-27T11:40:00Z",
                "inconsistency_count": 1
            },
            "config": {
                "consistent": False,
                "last_check": "2025-08-27T11:30:00Z",
                "inconsistency_count": 1
            }
        }
    },
    "fault_events": [
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
}

@app.get("/api/v1/fault-tolerance/status")
async def get_fault_tolerance_status():
    """获取故障容错系统状态"""
    return {
        "status": "healthy",
        "system_started": True,
        "active_faults": 0,
        "total_components": 5,
        "healthy_components": 5,
        "last_updated": datetime.now().isoformat()
    }

@app.get("/api/v1/fault-tolerance/recovery/statistics")
async def get_recovery_statistics():
    """获取恢复统计信息"""
    return fault_tolerance_stats["recovery_stats"]

@app.get("/api/v1/fault-tolerance/backup/statistics")
async def get_backup_statistics():
    """获取备份统计信息"""
    return fault_tolerance_stats["backup_stats"]

@app.get("/api/v1/fault-tolerance/consistency/statistics")
async def get_consistency_statistics():
    """获取一致性统计信息"""
    return fault_tolerance_stats["consistency_stats"]

@app.get("/api/v1/fault-tolerance/faults/events")
async def get_fault_events():
    """获取故障事件列表"""
    return {"events": fault_tolerance_stats["fault_events"]}

@app.post("/api/v1/fault-tolerance/testing/inject-fault")
async def inject_fault(fault_data: Dict[str, Any]):
    """注入故障测试"""
    fault_id = f"test_fault_{len(fault_tolerance_stats['fault_events']) + 1:03d}"
    new_fault = {
        "id": fault_id,
        "type": fault_data.get("fault_type", "test_fault"),
        "component": fault_data.get("component_id", "test_component"),
        "severity": fault_data.get("severity", "low"),
        "timestamp": datetime.now().isoformat(),
        "status": "injected"
    }
    fault_tolerance_stats["fault_events"].append(new_fault)
    return {
        "success": True,
        "fault_id": fault_id,
        "message": "故障注入成功"
    }

# 共情响应生成器 API (Story 11.3)
class EmpathyGenerateRequest(BaseModel):
    """共情响应生成请求"""
    user_id: str = Field(..., description="用户ID")
    message: str = Field(..., description="用户消息")
    emotion: Optional[str] = Field(None, description="情感类型")
    intensity: float = Field(0.5, ge=0.0, le=1.0, description="情感强度")
    preferred_empathy_type: Optional[str] = Field(None, description="偏好的共情类型")
    cultural_context: Optional[str] = Field(None, description="文化背景")
    max_response_length: int = Field(200, ge=20, le=500, description="最大响应长度")
    urgency_level: float = Field(0.5, ge=0.0, le=1.0, description="紧急程度")

class EmpathyResponseModel(BaseModel):
    """共情响应模型"""
    id: str
    response_text: str
    empathy_type: str
    emotion_addressed: str
    comfort_level: float
    personalization_score: float
    suggested_actions: List[str]
    tone: str
    confidence: float
    timestamp: str
    generation_time_ms: float
    cultural_adaptation: Optional[str]
    template_used: Optional[str]
    metadata: Dict[str, Any]

# 模拟的共情响应模板
empathy_templates = {
    "cognitive": {
        "sad": [
            "我理解你现在感到伤心，这种情绪是完全正常的。让我们一起分析一下这个情况。",
            "从你的描述中，我能感受到你的难过。这种感受是可以理解的，让我帮你梳理一下思路。",
        ],
        "angry": [
            "我明白你现在很愤怒，这种情绪反应是可以理解的。让我们客观地看待这个问题。",
            "你的愤怒情绪我能够理解。让我们冷静地分析一下导致这种情况的原因。",
        ],
        "anxious": [
            "我理解你的焦虑，这是面对不确定性时的正常反应。让我们一起理性地分析这个问题。",
            "你的担忧是可以理解的。让我帮你分析一下实际的风险和可能的解决方案。",
        ]
    },
    "affective": {
        "sad": [
            "我能深深感受到你的痛苦，你不是一个人在承受这些。我在这里陪伴着你。",
            "看到你这么难过，我的心也很痛。请知道我真心关心你，愿意与你分担这份痛苦。",
        ],
        "angry": [
            "我能感受到你内心的愤怒和不平，这种强烈的情感完全可以理解。",
            "你的愤怒我感同身受，遇到这样的情况任何人都会生气。",
        ],
        "anxious": [
            "我能感受到你内心的不安和担忧，这种焦虑的感觉真的很难受。",
            "你的焦虑我完全能够体会，这种担心和不安的感觉确实让人很痛苦。",
        ]
    },
    "compassionate": {
        "sad": [
            "我真心希望能够减轻你的痛苦。让我为你提供一些实际的帮助和支持。",
            "看到你这么痛苦，我想尽我所能来帮助你。让我们一起找到走出困境的方法。",
        ],
        "angry": [
            "我理解你的愤怒，也希望能帮助你找到处理这种情绪的方法。让我们一起寻找解决方案。",
            "你的愤怒是正当的，现在让我帮你找到一些实际的方法来处理这种情况。",
        ],
        "anxious": [
            "我想帮助你减轻这种焦虑感。让我为你提供一些实用的建议和支持。",
            "你的担忧是有道理的，让我帮你制定一些实际的行动计划来缓解这种焦虑。",
        ]
    }
}

suggested_actions_map = {
    "sad": ["深呼吸练习", "与信任的朋友交流", "写日记记录情感", "寻求专业帮助"],
    "angry": ["数到十冷静下来", "进行体育锻炼", "表达你的感受", "寻找解决问题的方法"],
    "anxious": ["练习正念冥想", "制定行动计划", "与他人分享担忧", "关注可以控制的事情"]
}

@app.post("/api/v1/empathy/generate", response_model=EmpathyResponseModel)
async def generate_empathy_response(request: EmpathyGenerateRequest):
    """
    生成共情响应
    
    生成基于用户情感状态和个性特征的个性化共情响应。
    """
    try:
        start_time = datetime.now()
        
        # 确定情感和共情类型
        emotion = request.emotion or "sad"
        empathy_type = request.preferred_empathy_type or random.choice(["cognitive", "affective", "compassionate"])
        
        # 选择合适的模板
        templates = empathy_templates.get(empathy_type, {}).get(emotion, empathy_templates["cognitive"]["sad"])
        response_text = random.choice(templates)
        
        # 文化适应
        cultural_adaptation = None
        if request.cultural_context:
            if request.cultural_context == "collectivist":
                response_text += " 记住，你的家人和朋友都在支持你。"
                cultural_adaptation = "collectivist_adapted"
            elif request.cultural_context == "individualist":
                response_text += " 相信自己的能力，你有力量克服这个困难。"
                cultural_adaptation = "individualist_adapted"
        
        # 生成建议行动
        suggested_actions = random.sample(
            suggested_actions_map.get(emotion, suggested_actions_map["sad"]), 
            k=random.randint(2, 3)
        )
        
        # 计算各种指标
        comfort_level = random.uniform(0.7, 0.95)
        personalization_score = random.uniform(0.6, 0.9)
        confidence = random.uniform(0.8, 0.98)
        
        # 确定语调
        tone = "gentle" if empathy_type == "affective" else "supportive" if empathy_type == "compassionate" else "understanding"
        
        # 生成时间
        end_time = datetime.now()
        generation_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # 生成响应
        response = EmpathyResponseModel(
            id=f"empathy_{uuid.uuid4().hex[:8]}",
            response_text=response_text,
            empathy_type=empathy_type,
            emotion_addressed=emotion,
            comfort_level=comfort_level,
            personalization_score=personalization_score,
            suggested_actions=suggested_actions,
            tone=tone,
            confidence=confidence,
            timestamp=end_time.isoformat(),
            generation_time_ms=generation_time_ms,
            cultural_adaptation=cultural_adaptation,
            template_used=f"{empathy_type}_{emotion}",
            metadata={
                "user_id": request.user_id,
                "message_length": len(request.message),
                "intensity": request.intensity,
                "urgency_level": request.urgency_level
            }
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate empathy response: {str(e)}"
        )

@app.get("/api/v1/empathy/strategies")
async def get_empathy_strategies():
    """
    获取可用的共情策略信息
    
    返回系统支持的共情策略类型和适用情况。
    """
    strategies_info = {
        "available_strategies": [
            {
                "type": "cognitive",
                "name": "认知共情",
                "description": "理解和识别情感，提供理性的共情回应",
                "suitable_for": ["分析性思维", "低情感强度", "理性处理"]
            },
            {
                "type": "affective",
                "name": "情感共情", 
                "description": "分享和镜像情感，提供情感上的共鸣",
                "suitable_for": ["高情感强度", "情感表达", "情感连接"]
            },
            {
                "type": "compassionate",
                "name": "慈悲共情",
                "description": "提供支持行动，情感安慰和建设性帮助",
                "suitable_for": ["困难情感", "危机情况", "需要支持"]
            }
        ],
        "cultural_contexts": ["collectivist", "individualist", "high_context", "low_context"],
        "supported_emotions": ["sad", "angry", "anxious", "happy", "surprised", "fearful"]
    }
    
    return strategies_info

@app.get("/api/v1/empathy/analytics")
async def get_empathy_analytics(
    user_id: Optional[str] = None
):
    """
    获取共情响应分析数据
    
    返回系统性能统计和用户个性化分析。
    """
    analytics_data = {
        "system_performance": {
            "total_responses_generated": random.randint(100, 1000),
            "average_response_time": random.uniform(200, 500),
            "success_rate": random.uniform(0.95, 0.99),
            "average_comfort_level": random.uniform(0.8, 0.9),
            "strategy_usage": {
                "cognitive": random.uniform(0.3, 0.4),
                "affective": random.uniform(0.3, 0.4),
                "compassionate": random.uniform(0.2, 0.4)
            }
        },
        "context_statistics": {
            "active_contexts": random.randint(5, 20),
            "average_conversation_length": random.uniform(3, 8),
            "context_retention_rate": random.uniform(0.85, 0.95)
        },
        "personalization_stats": {
            "users_with_profiles": random.randint(50, 200),
            "average_personalization_score": random.uniform(0.7, 0.9),
            "cultural_adaptation_usage": random.uniform(0.6, 0.8)
        }
    }
    
    # 如果指定用户，添加用户特定分析
    if user_id:
        analytics_data["user_analysis"] = {
            "user_id": user_id,
            "conversation_count": random.randint(1, 20),
            "favorite_empathy_type": random.choice(["cognitive", "affective", "compassionate"]),
            "average_comfort_received": random.uniform(0.7, 0.95),
            "most_common_emotion": random.choice(["sad", "anxious", "happy"])
        }
    
    return analytics_data

class EmpathyFeedbackRequest(BaseModel):
    """共惃反馈请求"""
    response_id: str
    rating: float = Field(..., ge=0.0, le=5.0, description="评分 (0-5)")
    feedback_text: Optional[str] = None
    user_id: Optional[str] = None

@app.post("/api/v1/empathy/feedback")
async def submit_empathy_feedback(request: EmpathyFeedbackRequest):
    """
    提交共情响应反馈
    
    用户可以对生成的共情响应进行评分和反馈，用于改进系统。
    """
    feedback_data = {
        "response_id": request.response_id,
        "rating": request.rating,
        "feedback_text": request.feedback_text,
        "user_id": request.user_id,
        "timestamp": datetime.now().isoformat(),
        "source": "api"
    }
    
    return {
        "message": "Feedback submitted successfully",
        "feedback_id": f"feedback_{int(datetime.now().timestamp())}",
        "status": "processed",
        "data": feedback_data
    }

@app.get("/api/v1/empathy/health")
async def empathy_health_check():
    """
    共情系统健康检查
    
    检查共情响应生成系统的健康状态。
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "details": {
            "healthy": True,
            "response_engine_status": "active",
            "template_count": len(empathy_templates),
            "supported_emotions": len(suggested_actions_map),
            "average_response_time": random.uniform(200, 400)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
