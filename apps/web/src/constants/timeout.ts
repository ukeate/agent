/**
 * 超时时间常量定义
 */

// WebSocket和对话超时时间常量（秒）
export const TIMEOUT_CONSTANTS = {
  // WebSocket和对话超时时间 - 30分钟
  WEBSOCKET_TIMEOUT_SECONDS: 1800,
  CONVERSATION_TIMEOUT_SECONDS: 1800,
  AGENT_RESPONSE_TIMEOUT_SECONDS: 1800,
  
  // 旧的超时时间（保留用于兼容性检查）
  LEGACY_WEBSOCKET_TIMEOUT_SECONDS: 300,
  LEGACY_AGENT_RESPONSE_TIMEOUT_SECONDS: 60,
} as const;

// 前端超时常量（毫秒）
export const FRONTEND_TIMEOUT_CONSTANTS = {
  // API客户端超时 - 30分钟
  API_CLIENT_TIMEOUT_MS: 1800000,
  
  // WebSocket相关超时
  WEBSOCKET_TIMEOUT_MS: 1800000,
  
  // 旧的超时时间（保留用于兼容性检查）
  LEGACY_API_CLIENT_TIMEOUT_MS: 30000,
} as const;

// 对话配置常量
export const CONVERSATION_CONSTANTS = {
  // 默认最大轮数
  DEFAULT_MAX_ROUNDS: 3,
  
  // 默认超时时间
  DEFAULT_TIMEOUT_SECONDS: TIMEOUT_CONSTANTS.CONVERSATION_TIMEOUT_SECONDS,
  
  // 默认自动回复
  DEFAULT_AUTO_REPLY: true,
} as const;

// 导出所有常量的联合类型
export type TimeoutConstants = typeof TIMEOUT_CONSTANTS;
export type FrontendTimeoutConstants = typeof FRONTEND_TIMEOUT_CONSTANTS;
export type ConversationConstants = typeof CONVERSATION_CONSTANTS;