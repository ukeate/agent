# Frontend Architecture

基于React 18.2+和选择的技术栈，以下是前端特定架构的详细设计：

## Component Architecture

### Component Organization
```text
src/
├── components/
│   ├── ui/                     # 通用UI组件
│   │   ├── Button/
│   │   ├── Input/
│   │   ├── Modal/
│   │   └── DataTable/
│   ├── layout/                 # 布局组件
│   │   ├── Header/
│   │   ├── Sidebar/
│   │   └── MainLayout/
│   ├── agent/                  # 智能体相关组件
│   │   ├── AgentCard/
│   │   ├── AgentConfig/
│   │   └── AgentStatus/
│   ├── conversation/           # 对话相关组件
│   │   ├── MessageList/
│   │   ├── MessageInput/
│   │   └── ConversationHeader/
│   ├── task/                   # 任务相关组件
│   │   ├── TaskDashboard/
│   │   ├── DAGVisualizer/
│   │   └── TaskProgress/
│   └── knowledge/              # 知识库组件
│       ├── SearchInterface/
│       ├── KnowledgeItem/
│       └── RAGResponse/
├── pages/                      # 页面组件
├── hooks/                      # 自定义hooks
├── services/                   # API服务层
├── stores/                     # 状态管理
├── utils/                      # 工具函数
└── types/                      # TypeScript类型定义
```

## State Management Architecture

### State Structure
```typescript
// stores/index.ts
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { AgentSlice, createAgentSlice } from './agentSlice';
import { ConversationSlice, createConversationSlice } from './conversationSlice';
import { TaskSlice, createTaskSlice } from './taskSlice';
import { AuthSlice, createAuthSlice } from './authSlice';
import { UISlice, createUISlice } from './uiSlice';

// 全局状态类型
export interface RootState extends
  AgentSlice,
  ConversationSlice,
  TaskSlice,
  AuthSlice,
  UISlice {}

// 创建根状态存储
export const useAppStore = create<RootState>()(
  devtools(
    persist(
      (...args) => ({
        ...createAgentSlice(...args),
        ...createConversationSlice(...args),
        ...createTaskSlice(...args),
        ...createAuthSlice(...args),
        ...createUISlice(...args),
      }),
      {
        name: 'ai-agent-store',
        partialize: (state) => ({
          // 只持久化必要的状态
          auth: state.auth,
          ui: {
            theme: state.ui.theme,
            sidebarCollapsed: state.ui.sidebarCollapsed,
          },
        }),
      }
    ),
    { name: 'ai-agent-store' }
  )
);
```

### State Management Patterns
- **分片模式**: 将状态按功能域分片，避免单一大状态对象
- **选择器模式**: 使用计算属性和记忆化选择器优化性能
- **乐观更新**: UI立即更新，API失败时回滚状态
- **错误边界**: 每个状态切片包含错误处理逻辑
- **持久化策略**: 仅持久化用户偏好和认证状态

## Routing Architecture

### Protected Route Pattern
```typescript
import React, { Suspense } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { Spin } from 'antd';
import { useAuthStore } from '@/stores/authStore';

interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredPermissions?: string[];
  fallbackPath?: string;
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  requiredPermissions = [],
  fallbackPath = '/login'
}) => {
  const location = useLocation();
  const { isAuthenticated, user, hasPermissions } = useAuthStore();

  // 检查认证状态
  if (!isAuthenticated) {
    return (
      <Navigate
        to={fallbackPath}
        state={{ from: location }}
        replace
      />
    );
  }

  // 检查权限
  if (requiredPermissions.length > 0 && !hasPermissions(requiredPermissions)) {
    return (
      <Navigate
        to="/unauthorized"
        state={{ from: location }}
        replace
      />
    );
  }

  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center h-64">
          <Spin size="large" tip="加载中..." />
        </div>
      }
    >
      {children}
    </Suspense>
  );
};
```

## Frontend Services Layer

### API Client Setup
```typescript
import axios, { AxiosInstance, AxiosError } from 'axios';
import { message } from 'antd';
import { useAuthStore } from '@/stores/authStore';

// API客户端配置
class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000/api/v1',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // 请求拦截器 - 添加认证头
    this.client.interceptors.request.use(
      (config) => {
        const { token } = useAuthStore.getState();
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // 响应拦截器 - 错误处理
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        this.handleError(error);
        return Promise.reject(error);
      }
    );
  }

  private handleError(error: AxiosError) {
    if (error.response?.status === 401) {
      // 未授权，清除认证状态
      useAuthStore.getState().logout();
      window.location.href = '/login';
      return;
    }

    if (error.response?.status === 403) {
      message.error('权限不足');
      return;
    }

    if (error.response?.status >= 500) {
      message.error('服务器错误，请稍后重试');
      return;
    }

    // 显示具体错误信息
    const errorMessage = error.response?.data?.message || error.message;
    message.error(errorMessage);
  }

  // 封装常用HTTP方法
  get<T = any>(url: string, params?: any): Promise<T> {
    return this.client.get(url, { params }).then(res => res.data);
  }

  post<T = any>(url: string, data?: any): Promise<T> {
    return this.client.post(url, data).then(res => res.data);
  }

  put<T = any>(url: string, data?: any): Promise<T> {
    return this.client.put(url, data).then(res => res.data);
  }

  delete<T = any>(url: string): Promise<T> {
    return this.client.delete(url).then(res => res.data);
  }

  // WebSocket连接管理
  createWebSocket(path: string): WebSocket {
    const wsUrl = process.env.REACT_APP_WS_BASE_URL || 'ws://localhost:8000';
    const { token } = useAuthStore.getState();
    return new WebSocket(`${wsUrl}${path}?token=${token}`);
  }
}

export const apiClient = new ApiClient();
```
