# External APIs

基于PRD要求和组件设计，项目需要集成以下外部服务来实现完整的AI功能：

## OpenAI API

- **Purpose:** 提供核心语言模型推理能力，支持多智能体对话和代码生成
- **Documentation:** https://platform.openai.com/docs/api-reference
- **Base URL(s):** https://api.openai.com/v1
- **Authentication:** API Key (Bearer Token)
- **Rate Limits:** 根据订阅计划，通常为每分钟500-10000请求

**Key Endpoints Used:**
- `POST /chat/completions` - 创建对话完成，支持工具调用和系统提示
- `POST /chat/completions` (stream=true) - 流式响应，实时生成内容
- `POST /embeddings` - 生成文本嵌入向量，支持RAG系统

**Integration Notes:** 需要实现重试机制和错误处理，支持工具调用格式转换，管理上下文长度限制

**Model Configuration:**
- **Primary Model:** gpt-4o-mini - 成本优化的高效推理模型
- **Embedding Model:** text-embedding-3-small - 轻量级向量化模型

**Integration Notes:** 批量处理优化，缓存常用嵌入向量，处理API限制和错误重试
