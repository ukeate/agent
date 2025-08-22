"""
简化版后端服务，仅用于测试异常检测功能
绕过复杂的初始化流程
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# 创建简化的应用
app = FastAPI(
    title="AI Agent System API (Simple)",
    version="0.1.0",
    docs_url="/docs"
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """根路径"""
    return {"message": "Simple AI Agent API", "status": "running"}

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}

# 添加异常检测路由
@app.get("/api/v1/analytics/anomalies")
async def get_anomalies(use_real_detection: bool = True):
    """异常检测接口"""
    try:
        if use_real_detection:
            # 延迟导入异常检测模块
            from ai.anomaly_detection import IntelligentAnomalyDetector, create_sample_events
            
            # 创建检测器和示例数据
            detector = IntelligentAnomalyDetector()
            events_data = create_sample_events(num_users=20, num_events=100)
            
            # 运行异常检测
            anomalies = detector.detect_anomalies(events_data, time_window=3600)
            
            # 转换为API响应格式
            result_anomalies = []
            for anomaly in anomalies[:10]:  # 限制返回数量
                result_anomalies.append({
                    "anomaly_id": anomaly.anomaly_id,
                    "user_id": anomaly.user_id,
                    "event_type": anomaly.event_type,
                    "timestamp": anomaly.timestamp.isoformat(),
                    "severity": anomaly.severity,
                    "confidence": anomaly.confidence,
                    "description": anomaly.description,
                    "anomaly_type": anomaly.anomaly_type,
                    "score": anomaly.score
                })
            
            # 生成摘要
            severity_counts = {}
            for anomaly in anomalies:
                severity_counts[anomaly.severity] = severity_counts.get(anomaly.severity, 0) + 1
            
            summary = {
                "total_anomalies": len(anomalies),
                "severity_distribution": severity_counts,
                "detection_algorithm": "IntelligentAnomalyDetector",
                "time_window": 3600,
                "users_analyzed": len(set(a.user_id for a in anomalies))
            }
            
            return {
                "success": True,
                "message": "智能异常检测完成",
                "summary": summary,
                "anomalies": result_anomalies,
                "algorithm_info": {
                    "version": "1.0",
                    "methods": ["统计异常检测", "机器学习检测", "行为特征分析"]
                }
            }
        else:
            # 返回演示数据
            return {
                "success": True,
                "message": "使用演示数据",
                "summary": {
                    "total_anomalies": 3,
                    "severity_distribution": {"high": 1, "medium": 2},
                    "detection_algorithm": "DemoDetector"
                },
                "anomalies": [
                    {
                        "anomaly_id": "demo_1",
                        "user_id": "user_1",
                        "event_type": "high_frequency",
                        "severity": "high",
                        "description": "演示：高频异常行为"
                    }
                ]
            }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "异常检测失败"
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)