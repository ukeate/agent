/**
 * 系统相关常量
 */

// 应用信息
export const APP_INFO = {
  NAME: 'AI Agent System',
  VERSION: '1.0.0',
  DESCRIPTION: '智能体协作系统',
  AUTHOR: 'Agent Development Team'
} as const;

// 默认配置
export const DEFAULT_CONFIG = {
  // 分页
  PAGINATION: {
    PAGE_SIZE: 20,
    MAX_PAGE_SIZE: 100,
    DEFAULT_PAGE: 1
  },
  
  // 智能体
  AGENT: {
    MAX_TOKENS: 4000,
    TEMPERATURE: 0.7,
    TIMEOUT: 30000,
    MAX_RETRY: 3
  },
  
  // 会话
  SESSION: {
    MAX_MESSAGES: 1000,
    CLEANUP_AFTER: 24 * 60 * 60 * 1000, // 24小时
    AUTO_SAVE_INTERVAL: 30000 // 30秒
  },
  
  // 文件上传
  UPLOAD: {
    MAX_SIZE: 10 * 1024 * 1024, // 10MB
    ALLOWED_TYPES: ['image/jpeg', 'image/png', 'image/gif', 'text/plain', 'application/pdf'],
    MAX_FILES: 5
  }
} as const;

// 环境变量键名
export const ENV_KEYS = {
  NODE_ENV: 'NODE_ENV',
  PORT: 'PORT',
  DATABASE_URL: 'DATABASE_URL',
  REDIS_URL: 'REDIS_URL',
  OPENAI_API_KEY: 'OPENAI_API_KEY',
  ANTHROPIC_API_KEY: 'ANTHROPIC_API_KEY',
  JWT_SECRET: 'JWT_SECRET',
  CORS_ORIGINS: 'CORS_ORIGINS'
} as const;

// 缓存键前缀
export const CACHE_KEYS = {
  SESSION: 'session:',
  AGENT: 'agent:',
  USER: 'user:',
  TASK: 'task:',
  HEALTH: 'health:',
  METRICS: 'metrics:'
} as const;

// 事件类型
export const EVENT_TYPES = {
  // 智能体事件
  AGENT_CREATED: 'agent.created',
  AGENT_UPDATED: 'agent.updated',
  AGENT_DELETED: 'agent.deleted',
  AGENT_STATUS_CHANGED: 'agent.status_changed',
  
  // 会话事件
  SESSION_CREATED: 'session.created',
  SESSION_UPDATED: 'session.updated',
  SESSION_ENDED: 'session.ended',
  MESSAGE_SENT: 'message.sent',
  MESSAGE_RECEIVED: 'message.received',
  
  // 任务事件
  TASK_CREATED: 'task.created',
  TASK_STARTED: 'task.started',
  TASK_COMPLETED: 'task.completed',
  TASK_FAILED: 'task.failed',
  
  // 系统事件
  SYSTEM_HEALTH_CHECK: 'system.health_check',
  SYSTEM_ERROR: 'system.error'
} as const;

// 错误代码
export const ERROR_CODES = {
  // 认证错误
  AUTH_REQUIRED: 'AUTH_REQUIRED',
  INVALID_TOKEN: 'INVALID_TOKEN',
  TOKEN_EXPIRED: 'TOKEN_EXPIRED',
  
  // 智能体错误
  AGENT_NOT_FOUND: 'AGENT_NOT_FOUND',
  AGENT_BUSY: 'AGENT_BUSY',
  AGENT_ERROR: 'AGENT_ERROR',
  
  // 会话错误
  SESSION_NOT_FOUND: 'SESSION_NOT_FOUND',
  SESSION_EXPIRED: 'SESSION_EXPIRED',
  MESSAGE_TOO_LONG: 'MESSAGE_TOO_LONG',
  
  // 系统错误
  INTERNAL_ERROR: 'INTERNAL_ERROR',
  SERVICE_UNAVAILABLE: 'SERVICE_UNAVAILABLE',
  RATE_LIMIT_EXCEEDED: 'RATE_LIMIT_EXCEEDED',
  
  // 验证错误
  VALIDATION_ERROR: 'VALIDATION_ERROR',
  INVALID_INPUT: 'INVALID_INPUT',
  MISSING_REQUIRED_FIELD: 'MISSING_REQUIRED_FIELD'
} as const;