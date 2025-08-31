"""
简单的测试API服务器，用于前端功能测试
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="AI Agent Test API", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "AI Agent Test API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# 知识图谱相关API
@app.get("/api/v1/knowledge/extraction/overview")
async def get_extraction_overview():
    return {
        "total_tasks": 156,
        "active_tasks": 8,
        "completed_tasks": 142,
        "failed_tasks": 6,
        "total_documents": 45320,
        "total_entities": 89456,
        "total_relations": 23789,
        "average_accuracy": 94.2
    }

@app.post("/api/v1/knowledge/entities/extract")
async def extract_entities(data: dict):
    return {
        "entities": [
            {"text": "张三", "label": "PERSON", "confidence": 0.95},
            {"text": "苹果公司", "label": "ORG", "confidence": 0.98}
        ]
    }

@app.post("/api/v1/knowledge/relations/extract")
async def extract_relations(data: dict):
    return {
        "relations": [
            {"subject": "张三", "predicate": "works_for", "object": "苹果公司", "confidence": 0.92}
        ]
    }

@app.get("/api/v1/agents")
async def get_agents():
    return {
        "agents": [
            {"id": "1", "name": "React Agent", "status": "active"},
            {"id": "2", "name": "Research Agent", "status": "active"}
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)