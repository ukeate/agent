/**
 * 通用类型定义
 */

// 基础ID类型
export type ID = string | number;

// 时间戳类型
export type Timestamp = string;

// 环境类型
export enum Environment {
  DEVELOPMENT = 'development',
  STAGING = 'staging',
  PRODUCTION = 'production'
}

// 日志级别
export enum LogLevel {
  DEBUG = 'debug',
  INFO = 'info',
  WARN = 'warn',
  ERROR = 'error',
  FATAL = 'fatal'
}

// 分页参数
export interface PaginationParams {
  page: number;
  pageSize: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

// 排序接口
export interface SortOptions {
  field: string;
  direction: 'asc' | 'desc';
}

// 过滤器接口
export interface FilterOptions {
  field: string;
  operator: 'eq' | 'ne' | 'gt' | 'gte' | 'lt' | 'lte' | 'in' | 'like';
  value: unknown;
}

// 搜索参数
export interface SearchParams {
  query?: string;
  filters?: FilterOptions[];
  sort?: SortOptions[];
  pagination?: PaginationParams;
}

// 键值对类型
export type KeyValue<T = unknown> = Record<string, T>;

// 选项接口
export interface Option<T = unknown> {
  label: string;
  value: T;
  disabled?: boolean;
}

// 文件信息
export interface FileInfo {
  name: string;
  size: number;
  type: string;
  lastModified: number;
  path?: string;
}

// 健康检查状态
export enum HealthStatus {
  HEALTHY = 'healthy',
  UNHEALTHY = 'unhealthy',
  DEGRADED = 'degraded'
}

// 健康检查结果
export interface HealthCheck {
  status: HealthStatus;
  checks: Record<string, {
    status: HealthStatus;
    message?: string;
    timestamp: Timestamp;
  }>;
  uptime: number;
  version: string;
}
