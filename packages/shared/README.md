# @agent/shared

AI Agent System 共享工具库和类型定义。

## 概述

这个包包含了前后端共享的类型定义、常量、工具函数和API客户端，确保跨项目的代码一致性和复用性。

## 安装

```bash
npm install @agent/shared
```

## 使用

### 类型定义

```typescript
import { Agent, Message, ApiResponse } from '@agent/shared';

const agent: Agent = {
  id: 'agent-1',
  config: { /* ... */ },
  status: AgentStatus.IDLE,
  // ...
};
```

### 常量

```typescript
import { API_ENDPOINTS, HTTP_STATUS } from '@agent/shared';

const endpoint = API_ENDPOINTS.AGENTS.LIST;
const status = HTTP_STATUS.OK;
```

### 工具函数

```typescript
import { 
  isValidEmail, 
  formatFileSize, 
  generateUUID,
  debounce 
} from '@agent/shared';

// 验证
const valid = isValidEmail('user@example.com');

// 格式化
const size = formatFileSize(1024000); // "1.02 MB"

// 生成UUID
const id = generateUUID();

// 防抖
const debouncedFn = debounce(() => console.log('Hello'), 300);
```

### API客户端

```typescript
import { createAgentApiClient } from '@agent/shared';

const apiClient = createAgentApiClient('http://localhost:8000/api/v1');

// 获取智能体列表
const agents = await apiClient.getAgents();

// 发送消息
const response = await apiClient.sendMessage('session-1', 'Hello');
```

## 目录结构

```
src/
├── types/          # 类型定义
│   ├── api.ts      # API相关类型
│   ├── agent.ts    # 智能体相关类型
│   ├── common.ts   # 通用类型
│   └── index.ts    # 类型导出
├── constants/      # 常量定义
│   ├── api.ts      # API相关常量
│   ├── system.ts   # 系统相关常量
│   └── index.ts    # 常量导出
├── utils/          # 工具函数
│   ├── validation.ts # 验证工具
│   ├── format.ts   # 格式化工具
│   ├── helpers.ts  # 辅助工具
│   └── index.ts    # 工具导出
├── api-client/     # API客户端
│   ├── base.ts     # 基础客户端
│   ├── agent.ts    # 智能体客户端
│   └── index.ts    # 客户端导出
└── index.ts        # 主导出文件
```

## 开发

```bash
# 构建
npm run build

# 开发模式（监听变化）
npm run dev

# 代码检查
npm run lint

# 测试
npm run test

# 清理构建产物
npm run clean
```

## 特性

### 类型安全
- 完整的 TypeScript 类型定义
- 前后端类型一致性
- 严格的类型检查

### API客户端
- 基于 Fetch API 的现代客户端
- 自动重试机制
- 请求/响应拦截器
- 错误处理

### 工具函数
- 常用验证函数
- 格式化工具
- 通用辅助函数
- 性能优化工具（防抖、节流）

### 常量管理
- 集中的常量定义
- API端点管理
- 系统配置常量

## 最佳实践

1. **类型优先**: 总是从类型定义开始，确保前后端一致性
2. **常量复用**: 使用共享常量避免硬编码
3. **工具函数**: 复用工具函数提高代码质量
4. **API客户端**: 使用统一的API客户端确保一致的错误处理

## 贡献

1. 所有新增类型都应该添加完整的文档注释
2. 工具函数需要包含单元测试
3. API变更需要同时更新类型定义和常量
4. 遵循现有的代码风格和命名约定