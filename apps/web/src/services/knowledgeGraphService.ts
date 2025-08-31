/**
 * 知识图谱数据服务
 * 
 * 功能包括：
 * - 图谱数据获取和缓存机制
 * - 查询转换和结果处理逻辑
 * - 统计数据计算和聚合功能
 * - WebSocket支持实时图谱更新
 * - API错误处理和重试机制
 */

import axios, { AxiosResponse, AxiosError } from 'axios';
import type {
  GraphData,
  GraphNode,
  GraphEdge,
  GraphStats,
  QueryHighlight,
  NLQuery,
  QueryResult,
  PathFindingConfig,
  NeighborhoodConfig,
  FilterConfig,
  SubgraphConfig,
  VisualizationConfig
} from '../components/knowledge-graph/GraphVisualization';

// ==================== 基础配置 ====================

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const API_TIMEOUT = 30000;
const CACHE_TTL = 5 * 60 * 1000; // 5分钟缓存
const MAX_RETRY_COUNT = 3;
const RETRY_DELAY = 1000; // 1秒

// ==================== API客户端配置 ====================

const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api/v1/knowledge-graph`,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  }
});

// ==================== 请求响应拦截器 ====================

apiClient.interceptors.request.use(
  (config) => {
    // 添加认证信息
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    // 添加请求时间戳
    config.metadata = { startTime: Date.now() };
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

apiClient.interceptors.response.use(
  (response) => {
    // 计算响应时间
    const endTime = Date.now();
    const duration = endTime - (response.config.metadata?.startTime || endTime);
    response.metadata = { duration };
    
    return response;
  },
  (error: AxiosError) => {
    console.error('Knowledge Graph API Error:', {
      url: error.config?.url,
      method: error.config?.method,
      status: error.response?.status,
      statusText: error.response?.statusText,
      data: error.response?.data
    });
    
    return Promise.reject(error);
  }
);

// ==================== 缓存管理 ====================

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number;
}

class CacheManager {
  private cache = new Map<string, CacheEntry<any>>();

  set<T>(key: string, data: T, ttl: number = CACHE_TTL): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl
    });
  }

  get<T>(key: string): T | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    const isExpired = Date.now() - entry.timestamp > entry.ttl;
    if (isExpired) {
      this.cache.delete(key);
      return null;
    }

    return entry.data;
  }

  clear(): void {
    this.cache.clear();
  }

  delete(key: string): void {
    this.cache.delete(key);
  }
}

const cache = new CacheManager();

// ==================== 重试机制 ====================

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

async function withRetry<T>(
  operation: () => Promise<T>,
  maxRetries: number = MAX_RETRY_COUNT,
  retryDelay: number = RETRY_DELAY
): Promise<T> {
  let lastError: Error;
  
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error as Error;
      
      if (attempt === maxRetries) {
        break;
      }
      
      // 指数退避策略
      const currentDelay = retryDelay * Math.pow(2, attempt);
      await delay(currentDelay);
    }
  }
  
  throw lastError!;
}

// ==================== WebSocket连接管理 ====================

class WebSocketManager {
  private ws: WebSocket | null = null;
  private listeners = new Map<string, Set<Function>>();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 5000;

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    const wsUrl = `${API_BASE_URL.replace('http', 'ws')}/ws/knowledge-graph`;
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log('Knowledge Graph WebSocket connected');
      this.reconnectAttempts = 0;
    };

    this.ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        this.notifyListeners(message.type, message.data);
      } catch (error) {
        console.error('WebSocket message parse error:', error);
      }
    };

    this.ws.onclose = () => {
      console.log('Knowledge Graph WebSocket disconnected');
      this.scheduleReconnect();
    };

    this.ws.onerror = (error) => {
      console.error('Knowledge Graph WebSocket error:', error);
    };
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  subscribe(event: string, callback: Function): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }

  unsubscribe(event: string, callback: Function): void {
    this.listeners.get(event)?.delete(callback);
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max WebSocket reconnect attempts reached');
      return;
    }

    this.reconnectAttempts++;
    setTimeout(() => {
      this.connect();
    }, this.reconnectInterval * this.reconnectAttempts);
  }

  private notifyListeners(event: string, data: any): void {
    const listeners = this.listeners.get(event);
    if (listeners) {
      listeners.forEach(callback => callback(data));
    }
  }
}

const wsManager = new WebSocketManager();

// ==================== API接口类型 ====================

interface ApiResponse<T> {
  success: boolean;
  data: T;
  message?: string;
  error?: string;
  metadata?: {
    total?: number;
    page?: number;
    pageSize?: number;
    processingTime?: number;
  };
}

interface GraphDataRequest {
  maxNodes?: number;
  maxEdges?: number;
  entityTypes?: string[];
  relationTypes?: string[];
  includeMetadata?: boolean;
  format?: 'full' | 'lightweight';
}

interface SubgraphRequest {
  entities: string[];
  depth?: number;
  maxNodes?: number;
  includeConnections?: boolean;
  filters?: FilterConfig;
}

// ==================== 知识图谱服务类 ====================

export class KnowledgeGraphService {
  
  // ==================== 图谱数据获取 ====================
  
  async getVisualizationData(params: GraphDataRequest = {}): Promise<GraphData> {
    const cacheKey = `graph-data-${JSON.stringify(params)}`;
    const cachedData = cache.get<GraphData>(cacheKey);
    
    if (cachedData) {
      return cachedData;
    }

    const response = await withRetry(async () => {
      return await apiClient.get<ApiResponse<GraphData>>('/visualization/data', { params });
    });

    if (response.data.success) {
      cache.set(cacheKey, response.data.data);
      return response.data.data;
    }

    throw new Error(response.data.error || '获取图谱数据失败');
  }

  async getSubgraphData(request: SubgraphRequest): Promise<GraphData> {
    const response = await withRetry(async () => {
      return await apiClient.post<ApiResponse<GraphData>>('/visualization/subgraph', request);
    });

    if (response.data.success) {
      return response.data.data;
    }

    throw new Error(response.data.error || '获取子图数据失败');
  }

  // ==================== 自然语言查询 ====================
  
  async processNaturalLanguageQuery(query: NLQuery): Promise<QueryResult> {
    const response = await withRetry(async () => {
      return await apiClient.post<ApiResponse<QueryResult>>('/visualization/query', query);
    });

    if (response.data.success) {
      return response.data.data;
    }

    throw new Error(response.data.error || '查询处理失败');
  }

  // ==================== 探索功能 ====================
  
  async findPaths(config: PathFindingConfig): Promise<QueryResult> {
    const response = await withRetry(async () => {
      return await apiClient.post<ApiResponse<QueryResult>>('/exploration/path-finding', config);
    });

    if (response.data.success) {
      return response.data.data;
    }

    throw new Error(response.data.error || '路径查找失败');
  }

  async exploreNeighborhood(config: NeighborhoodConfig): Promise<QueryResult> {
    const response = await withRetry(async () => {
      return await apiClient.post<ApiResponse<QueryResult>>('/exploration/neighborhood', config);
    });

    if (response.data.success) {
      return response.data.data;
    }

    throw new Error(response.data.error || '邻域探索失败');
  }

  async matchPatterns(patterns: Record<string, any>): Promise<QueryResult> {
    const response = await withRetry(async () => {
      return await apiClient.post<ApiResponse<QueryResult>>('/exploration/pattern-match', patterns);
    });

    if (response.data.success) {
      return response.data.data;
    }

    throw new Error(response.data.error || '模式匹配失败');
  }

  // ==================== 统计数据 ====================
  
  async getGraphStats(timeRange?: [string, string]): Promise<GraphStats> {
    const cacheKey = `graph-stats-${timeRange ? timeRange.join('-') : 'current'}`;
    const cachedStats = cache.get<GraphStats>(cacheKey);
    
    if (cachedStats) {
      return cachedStats;
    }

    const params = timeRange ? { startDate: timeRange[0], endDate: timeRange[1] } : {};
    const response = await withRetry(async () => {
      return await apiClient.get<ApiResponse<GraphStats>>('/stats/summary', { params });
    });

    if (response.data.success) {
      // 统计数据缓存时间较短
      cache.set(cacheKey, response.data.data, 2 * 60 * 1000); // 2分钟
      return response.data.data;
    }

    throw new Error(response.data.error || '获取统计数据失败');
  }

  async getDistributionStats(): Promise<GraphStats['distributions']> {
    const cacheKey = 'distribution-stats';
    const cachedStats = cache.get<GraphStats['distributions']>(cacheKey);
    
    if (cachedStats) {
      return cachedStats;
    }

    const response = await withRetry(async () => {
      return await apiClient.get<ApiResponse<GraphStats['distributions']>>('/stats/distributions');
    });

    if (response.data.success) {
      cache.set(cacheKey, response.data.data);
      return response.data.data;
    }

    throw new Error(response.data.error || '获取分布统计失败');
  }

  async getQualityMetrics(): Promise<GraphStats['quality']> {
    const response = await withRetry(async () => {
      return await apiClient.get<ApiResponse<GraphStats['quality']>>('/stats/quality');
    });

    if (response.data.success) {
      return response.data.data;
    }

    throw new Error(response.data.error || '获取质量指标失败');
  }

  async getGrowthTrends(timeRange: [string, string]): Promise<GraphStats['growth']> {
    const response = await withRetry(async () => {
      return await apiClient.get<ApiResponse<GraphStats['growth']>>('/stats/growth', {
        params: { startDate: timeRange[0], endDate: timeRange[1] }
      });
    });

    if (response.data.success) {
      return response.data.data;
    }

    throw new Error(response.data.error || '获取增长趋势失败');
  }

  // ==================== 配置管理 ====================
  
  async getVisualizationConfig(): Promise<VisualizationConfig> {
    const cacheKey = 'visualization-config';
    const cachedConfig = cache.get<VisualizationConfig>(cacheKey);
    
    if (cachedConfig) {
      return cachedConfig;
    }

    const response = await withRetry(async () => {
      return await apiClient.get<ApiResponse<VisualizationConfig>>('/visualization/config');
    });

    if (response.data.success) {
      cache.set(cacheKey, response.data.data);
      return response.data.data;
    }

    throw new Error(response.data.error || '获取可视化配置失败');
  }

  async updateVisualizationConfig(config: Partial<VisualizationConfig>): Promise<VisualizationConfig> {
    const response = await withRetry(async () => {
      return await apiClient.put<ApiResponse<VisualizationConfig>>('/visualization/config', config);
    });

    if (response.data.success) {
      // 更新缓存
      cache.set('visualization-config', response.data.data);
      return response.data.data;
    }

    throw new Error(response.data.error || '更新可视化配置失败');
  }

  // ==================== 导出功能 ====================
  
  async exportGraphData(
    format: 'json' | 'gexf' | 'graphml' | 'csv',
    options: {
      entities?: string[];
      includeMetadata?: boolean;
      compressed?: boolean;
    } = {}
  ): Promise<Blob> {
    const response = await withRetry(async () => {
      return await apiClient.post('/export', 
        { format, ...options },
        { responseType: 'blob' }
      );
    });

    return new Blob([response.data]);
  }

  // ==================== 实时更新 ====================
  
  subscribeToUpdates(callback: (data: any) => void): void {
    wsManager.subscribe('graph-update', callback);
    wsManager.connect();
  }

  subscribeToStats(callback: (stats: Partial<GraphStats>) => void): void {
    wsManager.subscribe('stats-update', callback);
    wsManager.connect();
  }

  unsubscribeFromUpdates(callback: (data: any) => void): void {
    wsManager.unsubscribe('graph-update', callback);
  }

  unsubscribeFromStats(callback: (stats: Partial<GraphStats>) => void): void {
    wsManager.unsubscribe('stats-update', callback);
  }

  // ==================== 工具方法 ====================
  
  clearCache(): void {
    cache.clear();
  }

  invalidateCache(pattern?: string): void {
    if (!pattern) {
      cache.clear();
      return;
    }

    // 删除匹配模式的缓存项
    for (const [key] of cache['cache'].entries()) {
      if (key.includes(pattern)) {
        cache.delete(key);
      }
    }
  }

  async healthCheck(): Promise<{
    status: 'healthy' | 'unhealthy';
    responseTime: number;
    timestamp: string;
  }> {
    const startTime = Date.now();
    
    try {
      const response = await apiClient.get('/health');
      const responseTime = Date.now() - startTime;
      
      return {
        status: response.status === 200 ? 'healthy' : 'unhealthy',
        responseTime,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        responseTime: Date.now() - startTime,
        timestamp: new Date().toISOString()
      };
    }
  }

  // ==================== 错误处理辅助 ====================
  
  static isNetworkError(error: AxiosError): boolean {
    return error.code === 'NETWORK_ERROR' || 
           error.message === 'Network Error' ||
           !error.response;
  }

  static isTimeoutError(error: AxiosError): boolean {
    return error.code === 'ECONNABORTED' ||
           error.message.includes('timeout');
  }

  static isServerError(error: AxiosError): boolean {
    return error.response ? error.response.status >= 500 : false;
  }

  static getErrorMessage(error: any): string {
    if (error.response?.data?.error) {
      return error.response.data.error;
    }
    
    if (error.message) {
      return error.message;
    }
    
    return '未知错误';
  }
}

// ==================== 单例实例 ====================

export const knowledgeGraphService = new KnowledgeGraphService();

// ==================== 默认导出 ====================

export default knowledgeGraphService;