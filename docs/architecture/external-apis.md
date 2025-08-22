# External APIs

基于PRD要求和组件设计，项目需要集成以下外部服务来实现完整的AI功能：

## OpenAI API

- **Purpose:** 提供核心语言模型推理能力，支持多智能体对话和代码生成
- **Documentation:** https://platform.openai.com/docs/api-reference
- **Base URL(s):** https://api.openai.com/v1
- **Authentication:** API Key (Bearer Token)
- **Rate Limits:** 根据订阅计划，通常为每分钟50-1000请求

**Key Endpoints Used:**
- `POST /messages` - 创建对话完成，支持工具调用和系统提示
- `POST /messages/stream` - 流式响应，实时生成内容

**Integration Notes:** 需要实现重试机制和错误处理，支持工具调用格式转换，管理上下文长度限制

## OpenAI Embeddings API

- **Purpose:** 生成文本向量表示，支持RAG系统的语义检索功能
- **Documentation:** https://platform.openai.com/docs/api-reference/embeddings
- **Base URL(s):** https://api.openai.com/v1
- **Authentication:** API Key (Bearer Token)
- **Rate Limits:** 每分钟3000请求，每分钟1M tokens

**Key Endpoints Used:**
- `POST /embeddings` - 生成文本嵌入向量，使用text-embedding-3-small模型

**Integration Notes:** 批量处理优化，缓存常用嵌入向量，处理API限制和错误重试
