#!/usr/bin/env python3
"""
RAG系统诊断脚本
检查配置并验证各个组件的连接状态
"""

import sys
import os
import asyncio
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "apps" / "api" / "src"))

async def diagnose_rag_system():
    """诊断RAG系统各组件状态"""
    print("🔍 RAG系统诊断开始...")
    
    issues = []
    
    # 1. 检查环境变量配置
    print("\n1️⃣ 检查环境变量配置...")
    
    try:
        from core.config import get_settings
        settings = get_settings()
        
        # 检查OpenAI API Key
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your-openai-api-key-here":
            issues.append("❌ OPENAI_API_KEY未正确配置")
            print("   ❌ OPENAI_API_KEY: 未正确设置")
        else:
            print(f"   ✅ OPENAI_API_KEY: 已设置 (长度: {len(settings.OPENAI_API_KEY)})")
        
        # 检查Qdrant配置
        print(f"   ✅ QDRANT_HOST: {settings.QDRANT_HOST}")
        print(f"   ✅ QDRANT_PORT: {settings.QDRANT_PORT}")
        
        # 检查数据库配置
        print(f"   ✅ DATABASE_URL: {settings.DATABASE_URL[:50]}...")
        print(f"   ✅ REDIS_URL: {settings.REDIS_URL}")
        
    except Exception as e:
        issues.append(f"❌ 配置加载失败: {e}")
        print(f"   ❌ 配置加载失败: {e}")
    
    # 2. 检查Qdrant连接
    print("\n2️⃣ 检查Qdrant向量数据库连接...")
    try:
        from core.qdrant import get_qdrant_client
        client = get_qdrant_client()
        
        # 检查连接
        collections = client.get_collections()
        print(f"   ✅ Qdrant连接成功，发现 {len(collections.collections)} 个集合")
        
        for collection in collections.collections:
            info = client.get_collection(collection.name)
            print(f"   📁 集合: {collection.name}, 向量数量: {info.vectors_count}")
            
    except Exception as e:
        issues.append(f"❌ Qdrant连接失败: {e}")
        print(f"   ❌ Qdrant连接失败: {e}")
    
    # 3. 检查OpenAI API连接
    print("\n3️⃣ 检查OpenAI API连接...")
    try:
        from ai.rag.embeddings import embedding_service
        
        # 尝试生成一个简单的嵌入
        test_embedding = await embedding_service.embed_text("测试连接")
        print(f"   ✅ OpenAI API连接成功，嵌入向量维度: {len(test_embedding)}")
        
    except Exception as e:
        issues.append(f"❌ OpenAI API连接失败: {e}")
        print(f"   ❌ OpenAI API连接失败: {e}")
    
    # 4. 检查Redis连接
    print("\n4️⃣ 检查Redis连接...")
    try:
        from core.redis import get_redis
        redis = get_redis()
        
        # 测试连接
        await redis.ping()
        print("   ✅ Redis连接成功")
        
    except Exception as e:
        issues.append(f"❌ Redis连接失败: {e}")
        print(f"   ❌ Redis连接失败: {e}")
    
    # 5. 总结诊断结果
    print("\n📋 诊断结果总结:")
    if not issues:
        print("✅ 所有组件都运行正常！RAG系统可以正常使用。")
    else:
        print("⚠️ 发现以下问题需要解决:")
        for issue in issues:
            print(f"   {issue}")
        
        print("\n🔧 修复建议:")
        if any("OPENAI_API_KEY" in issue for issue in issues):
            print("   1. 设置有效的OpenAI API Key到.env文件:")
            print("      OPENAI_API_KEY=sk-your-actual-api-key-here")
        
        if any("Qdrant" in issue for issue in issues):
            print("   2. 启动Qdrant服务:")
            print("      cd infrastructure/docker && docker-compose up -d qdrant")
        
        if any("Redis" in issue for issue in issues):
            print("   3. 启动Redis服务:")
            print("      cd infrastructure/docker && docker-compose up -d redis")

if __name__ == "__main__":
    asyncio.run(diagnose_rag_system())