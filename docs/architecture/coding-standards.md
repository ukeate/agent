# Coding Standards

基于AI-First开发模式和多智能体系统特点，定义关键编码标准以防止常见错误：

## Critical Fullstack Rules

- **Type Sharing:** 所有数据类型必须在packages/shared中定义，前后端从统一位置导入，避免类型不一致导致的运行时错误
- **API Calls:** 禁止直接使用fetch或axios，必须通过service层统一调用，确保错误处理、重试逻辑和监控的一致性
- **Environment Variables:** 禁止直接访问process.env，必须通过config对象访问，确保环境变量的验证和类型安全
- **Error Handling:** 所有API路由必须使用标准错误处理器，确保错误格式一致和安全信息过滤
- **State Updates:** 禁止直接修改状态对象，必须使用不可变更新模式，避免状态同步问题
- **AI API Calls:** 所有AI服务调用必须包含超时、重试和降级机制，防止系统阻塞
- **Async Operations:** 异步操作必须正确处理Promise rejection，避免未捕获的Promise错误
- **Database Transactions:** 涉及多表操作必须使用事务，确保数据一致性
- **Cache Invalidation:** 缓存更新必须遵循write-through模式，避免数据不一致
- **WebSocket Connections:** WebSocket连接必须实现重连机制和心跳检测
- **File Operations:** 文件操作必须验证路径和权限，防止路径遍历攻击
- **Input Validation:** 所有用户输入必须在前后端都进行验证，实现纵深防御

## Naming Conventions

| Element | Frontend | Backend | Example |
|---------|----------|---------|---------|
| Components | PascalCase | - | `AgentConfigPanel.tsx` |
| Hooks | camelCase with 'use' | - | `useAgentWebSocket.ts` |
| API Routes | - | kebab-case | `/api/v1/agent-conversations` |
| Python Classes | - | PascalCase | `AgentOrchestrator` |
| Python Functions | - | snake_case | `create_multi_agent_conversation` |
| Database Tables | - | snake_case | `agent_conversations` |
| Constants | UPPER_SNAKE_CASE | UPPER_SNAKE_CASE | `MAX_CONVERSATION_LENGTH` |
