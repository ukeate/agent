/**
 * API相关常量
 */

// API版本
export const API_VERSION = 'v1';

// API基础路径
export const API_BASE_PATH = `/api/${API_VERSION}`;

// API端点
export const API_ENDPOINTS = {
  // 认证相关
  AUTH: {
    LOGIN: '/auth/login',
    LOGOUT: '/auth/logout',
    REFRESH: '/auth/refresh',
    PROFILE: '/auth/profile'
  },
  
  // 智能体相关
  AGENTS: {
    LIST: '/agents',
    DETAIL: '/agents/:id',
    CREATE: '/agents',
    UPDATE: '/agents/:id',
    DELETE: '/agents/:id',
    STATUS: '/agents/:id/status'
  },
  
  // 多智能体相关
  MULTI_AGENT: {
    BASE: '/multi-agent',
    CHAT: '/multi-agent/chat',
    WEBSOCKET: '/multi-agent/ws/:sessionId',
    SESSIONS: '/multi-agent/sessions',
    HEALTH: '/multi-agent/health'
  },
  
  // 任务相关
  TASKS: {
    LIST: '/tasks',
    DETAIL: '/tasks/:id',
    CREATE: '/tasks',
    UPDATE: '/tasks/:id',
    DELETE: '/tasks/:id'
  },
  
  // 系统相关
  SYSTEM: {
    HEALTH: '/health',
    METRICS: '/metrics',
    STATUS: '/status'
  }
} as const;

// HTTP状态码
export const HTTP_STATUS = {
  OK: 200,
  CREATED: 201,
  NO_CONTENT: 204,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  CONFLICT: 409,
  INTERNAL_SERVER_ERROR: 500,
  SERVICE_UNAVAILABLE: 503
} as const;

// 请求超时配置
export const TIMEOUT_CONFIG = {
  DEFAULT: 30000,      // 30秒
  SHORT: 5000,         // 5秒
  LONG: 60000,         // 1分钟
  WEBSOCKET: 300000    // 5分钟
} as const;

// 重试配置
export const RETRY_CONFIG = {
  MAX_RETRIES: 3,
  INITIAL_DELAY: 1000,
  MAX_DELAY: 10000,
  BACKOFF_FACTOR: 2
} as const;