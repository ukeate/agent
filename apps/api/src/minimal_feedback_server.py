#!/usr/bin/env python3
"""
最小化反馈系统测试服务器
"""

import sys
sys.path.append('/Users/runout/awork/code/my_git/agent/apps/api/src')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 直接导入反馈系统路由，避免其他依赖
import os
os.environ.setdefault('TESTING', 'true')

# 修改 feedback.py 的导入，使用相对导入避免依赖问题
app = FastAPI(title="反馈系统测试API", version="1.0.0")

# 添加CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 手动创建基本反馈路由
from fastapi import APIRouter
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

@app.get("/")
async def root():
    return {"message": "反馈系统最小化测试服务器"}

@app.get("/api/v1/feedback/overview")
async def get_feedback_overview(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """获取反馈系统概览"""
    return {
        "success": True,
        "data": {
            "total_feedbacks": 15420,
            "feedback_types": {
                "rating": 4520,
                "like": 3890,
                "click": 2850,
                "comment": 2120,
                "bookmark": 1580,
                "view": 460
            },
            "unique_users": 1240,
            "average_quality_score": 0.82,
            "top_items": [
                {"item_id": "item-001", "feedback_count": 245},
                {"item_id": "item-002", "feedback_count": 189},
                {"item_id": "item-003", "feedback_count": 167},
                {"item_id": "item-004", "feedback_count": 134},
                {"item_id": "item-005", "feedback_count": 98}
            ]
        }
    }

@app.get("/api/v1/feedback/metrics/realtime")
async def get_realtime_feedback_metrics():
    """获取实时反馈指标"""
    import random
    from datetime import datetime
    
    return {
        "success": True,
        "data": {
            "active_sessions": random.randint(50, 200),
            "events_per_minute": random.randint(100, 500),
            "buffer_status": {
                "pending_events": random.randint(0, 100),
                "processed_events": random.randint(1000, 5000),
                "failed_events": random.randint(0, 10),
                "buffer_utilization": round(random.uniform(0.1, 0.8), 2)
            },
            "processing_latency": round(random.uniform(20, 150), 1),
            "quality_score": round(random.uniform(0.75, 0.95), 3),
            "anomaly_count": random.randint(0, 5),
            "last_updated": utc_now().isoformat()
        }
    }

@app.get("/api/v1/feedback/analytics/user/{user_id}")
async def get_user_feedback_analytics(user_id: str):
    """获取用户反馈分析"""
    import random
    
    return {
        "success": True,
        "data": {
            "user_id": user_id,
            "total_feedbacks": random.randint(50, 500),
            "feedback_distribution": {
                "rating": random.randint(10, 100),
                "like": random.randint(20, 150),
                "click": random.randint(50, 200),
                "comment": random.randint(5, 50),
                "bookmark": random.randint(3, 30),
                "view": random.randint(100, 300)
            },
            "engagement_score": round(random.uniform(0.5, 0.95), 3),
            "preferences": {
                "categories": ["科技", "教育", "娱乐"],
                "activity_hours": [9, 14, 20, 22],
                "device_preference": "mobile"
            },
            "quality_metrics": {
                "consistency_score": round(random.uniform(0.7, 0.9), 3),
                "diversity_score": round(random.uniform(0.6, 0.8), 3),
                "authenticity_score": round(random.uniform(0.8, 0.95), 3)
            }
        }
    }

@app.get("/api/v1/feedback/analytics/item/{item_id}")
async def get_item_feedback_analytics(item_id: str):
    """获取推荐项反馈分析"""
    import random
    
    # 生成随机的反馈分布
    feedback_types = ["rating", "like", "dislike", "bookmark", "share", "comment", "click", "view", "dwell_time", "scroll_depth"]
    feedback_distribution = {}
    for feedback_type in feedback_types:
        feedback_distribution[feedback_type] = random.randint(5, 150)
    
    total_feedbacks = sum(feedback_distribution.values())
    
    return {
        "success": True,
        "data": {
            "item_id": item_id,
            "total_feedbacks": total_feedbacks,
            "average_rating": round(random.uniform(3.5, 4.8), 2) if total_feedbacks > 0 else None,
            "like_ratio": round(random.uniform(0.6, 0.9), 3),
            "engagement_metrics": {
                "click_through_rate": round(random.uniform(0.05, 0.25), 3),
                "dwell_time_avg": round(random.uniform(120, 600), 1),
                "scroll_depth_avg": round(random.uniform(0.4, 0.8), 3),
                "interaction_rate": round(random.uniform(0.1, 0.4), 3),
                "completion_rate": round(random.uniform(0.3, 0.7), 3),
                "bounce_rate": round(random.uniform(0.2, 0.5), 3)
            },
            "feedback_distribution": feedback_distribution
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("启动反馈系统最小化测试服务器...")
    uvicorn.run(app, host="0.0.0.0", port=8000)