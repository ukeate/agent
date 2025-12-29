"""
API文档生成和增强模块
"""

from typing import Dict, Any, List, Optional
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
import json
from ai_agent_sdk import AIAgentClient
from ai_agent_sdk.experiments import ExperimentBuilder

from src.core.logging import get_logger
logger = get_logger(__name__)

def generate_enhanced_openapi_schema(app: FastAPI) -> Dict[str, Any]:
    """生成增强的OpenAPI文档"""
    
    # 获取基础OpenAPI schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags
    )
    
    # 增强schemas
    enhanced_schema = enhance_openapi_schema(openapi_schema)
    
    return enhanced_schema

def enhance_openapi_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """增强OpenAPI schema"""
    
    # 添加服务器信息
    schema["servers"] = [
        {
            "url": "https://api.ai-agent.com",
            "description": "生产环境"
        },
        {
            "url": "https://staging-api.ai-agent.com", 
            "description": "测试环境"
        },
        {
            "url": "http://localhost:8000",
            "description": "本地开发环境"
        }
    ]
    
    # 添加安全方案
    schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT Bearer token认证"
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API密钥认证"
        }
    }
    
    # 全局安全要求
    schema["security"] = [
        {"BearerAuth": []},
        {"ApiKeyAuth": []}
    ]
    
    # 添加详细的错误响应schemas
    error_schemas = generate_error_schemas()
    schema["components"]["schemas"].update(error_schemas)
    
    # 增强路径文档
    enhance_paths_documentation(schema)
    
    # 添加示例
    add_request_response_examples(schema)
    
    return schema

def generate_error_schemas() -> Dict[str, Any]:
    """生成错误响应schemas"""
    return {
        "ErrorResponse": {
            "type": "object",
            "required": ["error", "timestamp"],
            "properties": {
                "error": {
                    "type": "object",
                    "required": ["code", "message"],
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "错误代码",
                            "enum": [
                                "INVALID_REQUEST",
                                "UNAUTHORIZED", 
                                "FORBIDDEN",
                                "NOT_FOUND",
                                "CONFLICT",
                                "RATE_LIMITED",
                                "INTERNAL_ERROR",
                                "SERVICE_UNAVAILABLE"
                            ]
                        },
                        "message": {
                            "type": "string",
                            "description": "错误描述"
                        },
                        "details": {
                            "type": "object",
                            "description": "详细错误信息",
                            "additionalProperties": True
                        }
                    }
                },
                "request_id": {
                    "type": "string",
                    "description": "请求ID，用于问题追踪"
                },
                "timestamp": {
                    "type": "string",
                    "format": "date-time",
                    "description": "错误发生时间"
                }
            },
            "example": {
                "error": {
                    "code": "INVALID_REQUEST",
                    "message": "Invalid experiment configuration",
                    "details": {
                        "field": "variants",
                        "reason": "Traffic percentages must sum to 100"
                    }
                },
                "request_id": "req_123456",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        },
        "ValidationError": {
            "type": "object",
            "properties": {
                "loc": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "错误字段路径"
                },
                "msg": {
                    "type": "string",
                    "description": "错误消息"
                },
                "type": {
                    "type": "string",
                    "description": "错误类型"
                }
            }
        }
    }

def enhance_paths_documentation(schema: Dict[str, Any]) -> None:
    """增强路径文档"""
    
    # 为所有路径添加通用响应
    common_responses = {
        "400": {
            "description": "请求参数无效",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                }
            }
        },
        "401": {
            "description": "未认证或认证失败",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                }
            }
        },
        "403": {
            "description": "无权限访问",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                }
            }
        },
        "429": {
            "description": "请求过于频繁",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                }
            },
            "headers": {
                "Retry-After": {
                    "description": "建议重试等待时间(秒)",
                    "schema": {"type": "integer"}
                }
            }
        },
        "500": {
            "description": "服务器内部错误",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                }
            }
        }
    }
    
    # 为每个路径添加通用响应
    for path_data in schema["paths"].values():
        for method_data in path_data.values():
            if isinstance(method_data, dict) and "responses" in method_data:
                # 只添加不存在的状态码
                for status_code, response_data in common_responses.items():
                    if status_code not in method_data["responses"]:
                        method_data["responses"][status_code] = response_data

def add_request_response_examples(schema: Dict[str, Any]) -> None:
    """添加请求响应示例"""
    
    examples = {
        # 智能体相关示例
        "/api/v1/agents": {
            "post": {
                "request_example": {
                    "name": "DataAnalyst",
                    "type": "react",
                    "description": "专业的数据分析智能体",
                    "tools": ["calculator", "database", "chart_generator"],
                    "model": "claude-3.5-sonnet",
                    "temperature": 0.7,
                    "max_iterations": 10,
                    "system_prompt": "你是一个专业的数据分析师..."
                },
                "response_example": {
                    "data": {
                        "id": "agent_abc123",
                        "name": "DataAnalyst",
                        "type": "react",
                        "status": "active",
                        "created_at": "2024-01-01T00:00:00Z",
                        "updated_at": "2024-01-01T00:00:00Z"
                    },
                    "message": "Agent created successfully",
                    "timestamp": "2024-01-01T00:00:00Z"
                }
            }
        },
        
        # 实验相关示例
        "/api/v1/experiments": {
            "post": {
                "request_example": {
                    "name": "首页改版测试",
                    "description": "测试新版首页的转化率提升效果",
                    "hypothesis": "新版首页将提高转化率15%",
                    "status": "draft",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                    "variants": [
                        {
                            "id": "control",
                            "name": "原版首页",
                            "description": "当前线上版本",
                            "traffic_percentage": 50,
                            "is_control": True,
                            "config": {
                                "template": "homepage_v1.html",
                                "css_theme": "default"
                            }
                        },
                        {
                            "id": "treatment",
                            "name": "新版首页",
                            "description": "优化后的版本",
                            "traffic_percentage": 50,
                            "is_control": False,
                            "config": {
                                "template": "homepage_v2.html",
                                "css_theme": "modern"
                            }
                        }
                    ],
                    "metrics": [
                        {
                            "name": "conversion_rate",
                            "type": "proportion",
                            "numerator_event": "purchase",
                            "denominator_event": "visit",
                            "description": "购买转化率"
                        },
                        {
                            "name": "average_order_value",
                            "type": "continuous", 
                            "event": "purchase",
                            "value_field": "order_amount",
                            "description": "平均订单金额"
                        }
                    ],
                    "targeting_rules": {
                        "include_users": [],
                        "exclude_users": ["test_user_1", "test_user_2"],
                        "user_attributes": {
                            "country": ["CN", "US"],
                            "user_type": ["premium", "enterprise"]
                        }
                    }
                },
                "response_example": {
                    "data": {
                        "id": "exp_xyz789",
                        "name": "首页改版测试",
                        "status": "draft",
                        "created_at": "2024-01-01T00:00:00Z",
                        "variants_count": 2,
                        "metrics_count": 2
                    },
                    "message": "Experiment created successfully",
                    "timestamp": "2024-01-01T00:00:00Z"
                }
            }
        },
        
        # 事件跟踪示例
        "/api/v1/events/track": {
            "post": {
                "request_example": {
                    "user_id": "user_456",
                    "event_type": "purchase",
                    "experiment_id": "exp_xyz789",
                    "variant_id": "treatment",
                    "value": 159.99,
                    "timestamp": "2024-01-01T12:30:00Z",
                    "properties": {
                        "product_id": "prod_123",
                        "category": "electronics",
                        "payment_method": "credit_card",
                        "discount_applied": True,
                        "discount_amount": 20.00
                    }
                },
                "response_example": {
                    "data": {
                        "event_id": "evt_def456",
                        "processed": True,
                        "experiment_id": "exp_xyz789",
                        "variant_id": "treatment"
                    },
                    "message": "Event tracked successfully",
                    "timestamp": "2024-01-01T12:30:01Z"
                }
            }
        }
    }
    
    # 将示例添加到schema中
    for path, methods in examples.items():
        if path in schema["paths"]:
            for method, method_examples in methods.items():
                if method in schema["paths"][path]:
                    # 添加请求示例
                    if "request_example" in method_examples:
                        request_body = schema["paths"][path][method].get("requestBody")
                        if request_body and "content" in request_body:
                            for content_type in request_body["content"]:
                                request_body["content"][content_type]["example"] = method_examples["request_example"]
                    
                    # 添加响应示例
                    if "response_example" in method_examples:
                        responses = schema["paths"][path][method].get("responses", {})
                        for status_code in ["200", "201"]:  # 成功响应
                            if status_code in responses and "content" in responses[status_code]:
                                for content_type in responses[status_code]["content"]:
                                    responses[status_code]["content"][content_type]["example"] = method_examples["response_example"]

def generate_sdk_code_samples() -> Dict[str, Any]:
    """生成SDK代码示例"""
    
    return {
        "python": {
            "create_agent": '''

client = AIAgentClient(api_key="your-api-key")

agent = client.agents.create(
    name="DataAnalyst",
    type="react",
    description="数据分析专家",
    tools=["calculator", "database"],
    model="claude-3.5-sonnet"
)

logger.info("Agent ID", agent_id=agent.id)
            ''',
            
            "execute_agent": '''
result = client.agents.execute(
    agent_id="agent_abc123",
    input="分析最近一周的销售数据",
    context={"user_id": "user_456"}
)

logger.info("Response", output=result.output)
            ''',
            
            "create_experiment": '''

experiment = (ExperimentBuilder()
    .set_name("首页改版测试")
    .add_variant("control", "原版", 50, is_control=True)
    .add_variant("treatment", "新版", 50)
    .add_metric("conversion_rate", "proportion",
                numerator_event="purchase",
                denominator_event="visit")
    .build()
)

exp = client.experiments.create(experiment)
            ''',
            
            "track_event": '''
client.events.track(
    user_id="user_456",
    event_type="purchase",
    experiment_id="exp_xyz789",
    variant_id="treatment",
    value=159.99,
    properties={"product_id": "prod_123"}
)
            '''
        },
        
        "javascript": {
            "create_agent": '''

const client = new AIAgentClient({
  apiKey: 'your-api-key'
});

const agent = await client.agents.create({
  name: 'DataAnalyst',
  type: 'react',
  description: '数据分析专家',
  tools: ['calculator', 'database'],
  model: 'claude-3.5-sonnet'
});

console.log('Agent ID:', agent.id);
            ''',
            
            "execute_agent": '''
const result = await client.agents.execute({
  agentId: 'agent_abc123',
  input: '分析最近一周的销售数据',
  context: { userId: 'user_456' }
});

console.log('Response:', result.output);
            ''',
            
            "create_experiment": '''
const experiment = await client.experiments.create({
  name: '首页改版测试',
  variants: [
    {
      id: 'control',
      name: '原版',
      traffic_percentage: 50,
      is_control: true
    },
    {
      id: 'treatment', 
      name: '新版',
      traffic_percentage: 50,
      is_control: false
    }
  ],
  metrics: [
    {
      name: 'conversion_rate',
      type: 'proportion',
      numerator_event: 'purchase',
      denominator_event: 'visit'
    }
  ]
});
            ''',
            
            "track_event": '''
await client.events.track({
  userId: 'user_456',
  eventType: 'purchase',
  experimentId: 'exp_xyz789',
  variantId: 'treatment',
  value: 159.99,
  properties: { productId: 'prod_123' }
});
            '''
        },
        
        "curl": {
            "create_agent": '''
curl -X POST "https://api.ai-agent.com/api/v1/agents" \\
  -H "Authorization: Bearer your-api-key" \\
  -H "Content-Type: application/json" \\
  -d '{
    "name": "DataAnalyst",
    "type": "react", 
    "description": "数据分析专家",
    "tools": ["calculator", "database"],
    "model": "claude-3.5-sonnet"
  }'
            ''',
            
            "execute_agent": '''
curl -X POST "https://api.ai-agent.com/api/v1/agents/agent_abc123/execute" \\
  -H "Authorization: Bearer your-api-key" \\
  -H "Content-Type: application/json" \\
  -d '{
    "input": "分析最近一周的销售数据",
    "context": {"user_id": "user_456"}
  }'
            ''',
            
            "track_event": '''
curl -X POST "https://api.ai-agent.com/api/v1/events/track" \\
  -H "Authorization: Bearer your-api-key" \\
  -H "Content-Type: application/json" \\
  -d '{
    "user_id": "user_456",
    "event_type": "purchase",
    "experiment_id": "exp_xyz789", 
    "variant_id": "treatment",
    "value": 159.99,
    "properties": {"product_id": "prod_123"}
  }'
            '''
        }
    }

def add_code_samples_to_schema(schema: Dict[str, Any]) -> None:
    """将代码示例添加到schema中"""
    
    code_samples = generate_sdk_code_samples()
    
    # 为路径添加代码示例
    path_mappings = {
        "/api/v1/agents": {
            "post": "create_agent"
        },
        "/api/v1/agents/{agent_id}/execute": {
            "post": "execute_agent" 
        },
        "/api/v1/experiments": {
            "post": "create_experiment"
        },
        "/api/v1/events/track": {
            "post": "track_event"
        }
    }
    
    for path, methods in path_mappings.items():
        if path in schema["paths"]:
            for method, sample_key in methods.items():
                if method in schema["paths"][path]:
                    # 添加x-code-samples扩展
                    schema["paths"][path][method]["x-code-samples"] = [
                        {
                            "lang": "Python",
                            "source": code_samples["python"][sample_key].strip()
                        },
                        {
                            "lang": "JavaScript", 
                            "source": code_samples["javascript"][sample_key].strip()
                        },
                        {
                            "lang": "cURL",
                            "source": code_samples["curl"][sample_key].strip()
                        }
                    ]

def generate_postman_collection(schema: Dict[str, Any]) -> Dict[str, Any]:
    """生成Postman集合"""
    
    collection = {
        "info": {
            "name": "AI Agent System API",
            "description": schema.get("info", {}).get("description", ""),
            "version": schema.get("info", {}).get("version", "1.0.0"),
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
        },
        "auth": {
            "type": "bearer",
            "bearer": [
                {
                    "key": "token",
                    "value": "{{api_token}}",
                    "type": "string"
                }
            ]
        },
        "variable": [
            {
                "key": "base_url",
                "value": "https://api.ai-agent.com",
                "type": "string"
            },
            {
                "key": "api_token",
                "value": "your-api-key-here",
                "type": "string"
            }
        ],
        "item": []
    }
    
    # 按标签组织请求
    tag_folders = {}
    
    for path, path_data in schema.get("paths", {}).items():
        for method, method_data in path_data.items():
            if not isinstance(method_data, dict):
                continue
                
            tags = method_data.get("tags", ["Default"])
            tag = tags[0] if tags else "Default"
            
            if tag not in tag_folders:
                tag_folders[tag] = {
                    "name": tag.title(),
                    "item": []
                }
            
            # 创建请求项
            request_item = create_postman_request_item(path, method, method_data)
            tag_folders[tag]["item"].append(request_item)
    
    collection["item"] = list(tag_folders.values())
    
    return collection

def create_postman_request_item(path: str, method: str, method_data: Dict[str, Any]) -> Dict[str, Any]:
    """创建Postman请求项"""
    
    # 处理路径参数
    postman_path = path.replace("{", "{{").replace("}", "}}")
    
    request_item = {
        "name": method_data.get("summary", f"{method.upper()} {path}"),
        "request": {
            "method": method.upper(),
            "header": [
                {
                    "key": "Content-Type",
                    "value": "application/json",
                    "type": "text"
                }
            ],
            "url": {
                "raw": f"{{{{base_url}}}}{postman_path}",
                "host": ["{{base_url}}"],
                "path": postman_path.strip("/").split("/")
            }
        },
        "response": []
    }
    
    # 添加请求体示例
    request_body = method_data.get("requestBody")
    if request_body and "content" in request_body:
        json_content = request_body["content"].get("application/json")
        if json_content and "example" in json_content:
            request_item["request"]["body"] = {
                "mode": "raw",
                "raw": json.dumps(json_content["example"], indent=2),
                "options": {
                    "raw": {
                        "language": "json"
                    }
                }
            }
    
    # 添加查询参数
    parameters = method_data.get("parameters", [])
    query_params = [p for p in parameters if p.get("in") == "query"]
    if query_params:
        request_item["request"]["url"]["query"] = [
            {
                "key": param["name"],
                "value": f"{{{{{param['name']}}}}}",
                "description": param.get("description", ""),
                "disabled": not param.get("required", False)
            }
            for param in query_params
        ]
    
    return request_item

def save_documentation_files(schema: Dict[str, Any], output_dir: str = "docs/api") -> None:
    """保存文档文件"""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存OpenAPI schema
    with open(f"{output_dir}/openapi.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    
    # 保存Postman集合
    postman_collection = generate_postman_collection(schema)
    with open(f"{output_dir}/postman_collection.json", "w", encoding="utf-8") as f:
        json.dump(postman_collection, f, indent=2, ensure_ascii=False)
    
    # 生成Insomnia集合
    insomnia_collection = generate_insomnia_collection(schema)
    with open(f"{output_dir}/insomnia_collection.json", "w", encoding="utf-8") as f:
        json.dump(insomnia_collection, f, indent=2, ensure_ascii=False)

def generate_insomnia_collection(schema: Dict[str, Any]) -> Dict[str, Any]:
    """生成Insomnia集合"""
    
    collection = {
        "_type": "export",
        "__export_format": 4,
        "__export_date": "2024-01-01T00:00:00.000Z",
        "__export_source": "ai-agent-system",
        "resources": [
            {
                "_id": "wrk_base",
                "_type": "workspace",
                "name": "AI Agent System API",
                "description": schema.get("info", {}).get("description", "")
            },
            {
                "_id": "env_base",
                "_type": "environment",
                "name": "Base Environment",
                "data": {
                    "base_url": "https://api.ai-agent.com",
                    "api_token": "your-api-key-here"
                }
            }
        ]
    }
    
    # 添加请求
    for path, path_data in schema.get("paths", {}).items():
        for method, method_data in path_data.items():
            if not isinstance(method_data, dict):
                continue
            
            request_id = f"req_{abs(hash(f'{method}_{path}'))}"
            
            request_item = {
                "_id": request_id,
                "_type": "request",
                "name": method_data.get("summary", f"{method.upper()} {path}"),
                "url": f"{{% base_url %}}{path}",
                "method": method.upper(),
                "headers": [
                    {
                        "name": "Authorization",
                        "value": "Bearer {% api_token %}"
                    },
                    {
                        "name": "Content-Type", 
                        "value": "application/json"
                    }
                ],
                "parentId": "wrk_base"
            }
            
            # 添加请求体
            request_body = method_data.get("requestBody")
            if request_body and "content" in request_body:
                json_content = request_body["content"].get("application/json")
                if json_content and "example" in json_content:
                    request_item["body"] = {
                        "mimeType": "application/json",
                        "text": json.dumps(json_content["example"], indent=2)
                    }
            
            collection["resources"].append(request_item)
    
    return collection

def generate_api_changelog() -> List[Dict[str, Any]]:
    """生成API变更日志"""
    
    return [
        {
            "version": "1.0.0",
            "date": "2024-01-01",
            "changes": [
                {
                    "type": "added",
                    "description": "初始API版本发布",
                    "details": [
                        "智能体管理接口",
                        "多智能体协作接口",
                        "工作流管理接口",
                        "RAG系统接口",
                        "A/B测试实验接口",
                        "事件跟踪接口",
                        "统计分析接口"
                    ]
                }
            ]
        }
    ]

# CLI工具用于生成文档
def main():
    """命令行工具入口"""
    import argparse
    from fastapi import FastAPI
    
    parser = argparse.ArgumentParser(description="生成API文档")
    parser.add_argument("--output", "-o", default="docs/api", help="输出目录")
    parser.add_argument("--format", choices=["json", "yaml"], default="json", help="输出格式")
    
    args = parser.parse_args()
    
    # 创建临时FastAPI应用以生成schema
    app = FastAPI(
        title="AI Agent System API",
        description="AI智能体系统平台API",
        version="1.0.0"
    )
    
    # 这里应该包含所有路由，简化起见省略
    
    # 生成增强的OpenAPI schema
    enhanced_schema = generate_enhanced_openapi_schema(app)
    add_code_samples_to_schema(enhanced_schema)
    
    # 保存文档文件
    save_documentation_files(enhanced_schema, args.output)
    
    logger.info("API文档已生成", output_path=args.output)
    logger.info("包含文件", files=["openapi.json", "postman_collection.json", "insomnia_collection.json"])

if __name__ == "__main__":
    main()
