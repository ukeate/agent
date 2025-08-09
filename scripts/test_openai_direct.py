#!/usr/bin/env python3
"""
直接测试OpenAI API连接
"""

import sys
import asyncio
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "apps" / "api" / "src"))

async def test_openai_connection():
    """测试OpenAI API连接"""
    print("🔍 测试OpenAI API直接连接...")
    
    try:
        from core.config import get_settings
        settings = get_settings()
        
        print(f"API Key前缀: {settings.OPENAI_API_KEY[:20]}...")
        print(f"API Key长度: {len(settings.OPENAI_API_KEY)}")
        
        # 直接使用OpenAI客户端测试
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        print("尝试调用embeddings API...")
        response = await client.embeddings.create(
            model="text-embedding-ada-002",
            input="Hello, world!"
        )
        
        embedding = response.data[0].embedding
        print(f"✅ OpenAI API连接成功！嵌入向量维度: {len(embedding)}")
        
    except Exception as e:
        print(f"❌ OpenAI API连接失败:")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        
        # 如果有详细的错误信息
        if hasattr(e, 'response'):
            print(f"   HTTP状态码: {e.response.status_code}")
            print(f"   响应内容: {e.response.text}")

if __name__ == "__main__":
    asyncio.run(test_openai_connection())