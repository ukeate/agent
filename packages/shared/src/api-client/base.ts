/**
 * 基础API客户端
 */

import type { ApiResponse, HttpMethod, RequestConfig } from '../types/api';
import { HTTP_STATUS, TIMEOUT_CONFIG, RETRY_CONFIG } from '../constants/api';

export interface ApiClientConfig {
  baseURL: string;
  timeout?: number;
  retries?: number;
  headers?: Record<string, string>;
  onRequest?: (config: RequestInit) => RequestInit | Promise<RequestInit>;
  onResponse?: (response: Response) => Response | Promise<Response>;
  onError?: (error: Error) => void;
}

export class ApiClient {
  private config: ApiClientConfig;

  constructor(config: ApiClientConfig) {
    this.config = {
      timeout: TIMEOUT_CONFIG.DEFAULT,
      retries: RETRY_CONFIG.MAX_RETRIES,
      ...config
    };
  }

  async request<T = any>(
    method: HttpMethod,
    endpoint: string,
    data?: any,
    options?: RequestConfig
  ): Promise<ApiResponse<T>> {
    const url = this.buildUrl(endpoint);
    const requestConfig = this.buildRequestConfig(method, data, options);

    let lastError: Error;
    for (let attempt = 0; attempt <= (options?.retries ?? this.config.retries ?? 0); attempt++) {
      try {
        const response = await this.executeRequest(url, requestConfig);
        return await this.processResponse<T>(response);
      } catch (error) {
        lastError = error as Error;
        
        // 不重试的错误类型
        if (this.shouldNotRetry(error as Error)) {
          throw error;
        }
        
        // 最后一次尝试失败
        if (attempt === (options?.retries ?? this.config.retries ?? 0)) {
          break;
        }
        
        // 等待后重试
        await this.sleep(RETRY_CONFIG.INITIAL_DELAY * Math.pow(RETRY_CONFIG.BACKOFF_FACTOR, attempt));
      }
    }

    if (this.config.onError) {
      this.config.onError(lastError!);
    }
    throw lastError!;
  }

  async get<T = any>(endpoint: string, options?: RequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>('GET', endpoint, undefined, options);
  }

  async post<T = any>(endpoint: string, data?: any, options?: RequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>('POST', endpoint, data, options);
  }

  async put<T = any>(endpoint: string, data?: any, options?: RequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>('PUT', endpoint, data, options);
  }

  async patch<T = any>(endpoint: string, data?: any, options?: RequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>('PATCH', endpoint, data, options);
  }

  async delete<T = any>(endpoint: string, options?: RequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>('DELETE', endpoint, undefined, options);
  }

  private buildUrl(endpoint: string): string {
    const baseURL = this.config.baseURL.replace(/\/$/, '');
    const path = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
    return `${baseURL}${path}`;
  }

  private buildRequestConfig(method: HttpMethod, data?: any, options?: RequestConfig): RequestInit {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...this.config.headers,
      ...options?.headers
    };

    const config: RequestInit = {
      method,
      headers,
      signal: this.createAbortSignal(options?.timeout ?? this.config.timeout)
    };

    if (data && method !== 'GET') {
      config.body = JSON.stringify(data);
    }

    return config;
  }

  private async executeRequest(url: string, config: RequestInit): Promise<Response> {
    // 请求拦截器
    if (this.config.onRequest) {
      config = await this.config.onRequest(config);
    }

    const response = await fetch(url, config);

    // 响应拦截器
    if (this.config.onResponse) {
      return await this.config.onResponse(response);
    }

    return response;
  }

  private async processResponse<T>(response: Response): Promise<ApiResponse<T>> {
    const contentType = response.headers.get('content-type');
    const isJson = contentType?.includes('application/json');

    let data: any;
    try {
      data = isJson ? await response.json() : await response.text();
    } catch (error) {
      throw new Error('响应解析失败');
    }

    if (!response.ok) {
      throw new ApiError(
        data.error || `HTTP ${response.status}: ${response.statusText}`,
        response.status,
        data
      );
    }

    return {
      success: true,
      data,
      timestamp: new Date().toISOString()
    };
  }

  private createAbortSignal(timeout?: number): AbortSignal {
    if (!timeout) {
      return new AbortController().signal;
    }

    const controller = new AbortController();
    setTimeout(() => controller.abort(), timeout);
    return controller.signal;
  }

  private shouldNotRetry(error: Error): boolean {
    if (error instanceof ApiError) {
      // 4xx错误通常不需要重试
      return error.status >= 400 && error.status < 500;
    }
    return false;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

export class ApiError extends Error {
  public status: number;
  public data?: any;

  constructor(message: string, status: number, data?: any) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}