import apiClient from './apiClient';

// ==================== 类型定义 ====================

export interface LoginRequest {
  username: string;
  password: string;
  scope?: string[];
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
  full_name?: string;
}

export interface Token {
  access_token: string;
  token_type: string;
  expires_in: number;
  refresh_token?: string;
  scope?: string;
}

export interface User {
  id: string;
  username: string;
  email: string;
  full_name?: string;
  is_active: boolean;
  is_superuser: boolean;
  roles: string[];
  permissions: string[];
  created_at: string;
  last_login?: string;
}

export interface PasswordChangeRequest {
  current_password: string;
  new_password: string;
}

// ==================== Service Class ====================

class AuthService {
  private baseUrl = '/auth';
  private tokenKey = 'auth_token';
  private refreshTokenKey = 'refresh_token';
  private userKey = 'current_user';

  constructor() {
    const token = this.getToken();
    if (token) {
      apiClient.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    }
  }

  // ==================== 认证管理 ====================

  async login(credentials: LoginRequest): Promise<Token> {
    const formData = new FormData();
    formData.append('username', credentials.username);
    formData.append('password', credentials.password);
    if (credentials.scope) {
      credentials.scope.forEach(s => formData.append('scope', s));
    }

    const response = await apiClient.post(`${this.baseUrl}/token`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    
    const token = response.data;
    this.saveToken(token);
    return token;
  }

  async logout(): Promise<void> {
    try {
      await apiClient.post(`${this.baseUrl}/logout`);
    } finally {
      this.clearAuth();
    }
  }

  async refreshToken(): Promise<Token> {
    const refreshToken = this.getRefreshToken();
    if (!refreshToken) {
      throw new Error('No refresh token available');
    }

    const response = await apiClient.post(`${this.baseUrl}/refresh`, {
      refresh_token: refreshToken
    });
    
    const token = response.data;
    this.saveToken(token);
    return token;
  }

  async register(userData: RegisterRequest): Promise<User> {
    const response = await apiClient.post(`${this.baseUrl}/register`, userData);
    return response.data;
  }

  // ==================== 用户管理 ====================

  async getCurrentUser(): Promise<User> {
    // 先尝试从缓存获取
    const cachedUser = this.getCachedUser();
    if (cachedUser) {
      return cachedUser;
    }

    // 从API获取
    const response = await apiClient.get(`${this.baseUrl}/me`);
    const user = response.data;
    this.saveUser(user);
    return user;
  }

  async updateProfile(updates: Partial<User>): Promise<User> {
    const response = await apiClient.put(`${this.baseUrl}/me`, updates);
    const user = response.data;
    this.saveUser(user);
    return user;
  }

  async changePassword(request: PasswordChangeRequest): Promise<{ message: string }> {
    const response = await apiClient.post(`${this.baseUrl}/change-password`, request);
    return response.data;
  }

  // ==================== 权限管理 ====================

  async checkPermission(resource: string, action: string): Promise<boolean> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/check-permission`, {
        resource,
        action
      });
      return response.data.has_permission;
    } catch (error) {
      return false;
    }
  }

  async getPermissionSummary(): Promise<{
    user_id: string;
    username: string;
    roles: string[];
    permissions: string[];
  }> {
    const response = await apiClient.get(`${this.baseUrl}/permissions`);
    return response.data;
  }

  async verifyToken(): Promise<{
    valid: boolean;
    user_id: string;
    username: string;
    expires_at: string | null;
    permissions: string[];
    roles: string[];
  }> {
    const response = await apiClient.get(`${this.baseUrl}/verify`);
    return response.data;
  }

  // ==================== Token管理 ====================

  getToken(): string | null {
    return localStorage.getItem(this.tokenKey);
  }

  getRefreshToken(): string | null {
    return localStorage.getItem(this.refreshTokenKey);
  }

  isAuthenticated(): boolean {
    const token = this.getToken();
    if (!token) return false;

    try {
      // 解析JWT token检查是否过期
      const payload = JSON.parse(atob(token.split('.')[1]));
      const exp = payload.exp * 1000; // 转换为毫秒
      return Date.now() < exp;
    } catch (error) {
      return false;
    }
  }

  private saveToken(token: Token): void {
    localStorage.setItem(this.tokenKey, token.access_token);
    if (token.refresh_token) {
      localStorage.setItem(this.refreshTokenKey, token.refresh_token);
    }
    
    // 设置axios默认header
    apiClient.defaults.headers.common['Authorization'] = `Bearer ${token.access_token}`;
  }

  private clearAuth(): void {
    localStorage.removeItem(this.tokenKey);
    localStorage.removeItem(this.refreshTokenKey);
    localStorage.removeItem(this.userKey);
    delete apiClient.defaults.headers.common['Authorization'];
  }

  private saveUser(user: User): void {
    localStorage.setItem(this.userKey, JSON.stringify(user));
  }

  private getCachedUser(): User | null {
    const userStr = localStorage.getItem(this.userKey);
    if (!userStr) return null;
    
    try {
      return JSON.parse(userStr);
    } catch (error) {
      return null;
    }
  }

  // ==================== 会话管理 ====================

  async validateSession(): Promise<boolean> {
    try {
      await this.getCurrentUser();
      return true;
    } catch (error) {
      // Token可能过期，尝试刷新
      try {
        await this.refreshToken();
        return true;
      } catch (refreshError) {
        this.clearAuth();
        return false;
      }
    }
  }

  async extendSession(): Promise<Token> {
    return this.refreshToken();
  }

  // ==================== 安全功能 ====================

  async getLoginHistory(limit: number = 10): Promise<Array<{
    timestamp: string;
    ip_address: string;
    user_agent: string;
    location?: string;
    success: boolean;
  }>> {
    const response = await apiClient.get(`${this.baseUrl}/login-history`, {
      params: { limit }
    });
    return response.data;
  }

  async getActiveSessions(): Promise<Array<{
    session_id: string;
    ip_address: string;
    user_agent: string;
    created_at: string;
    last_activity: string;
    is_current: boolean;
  }>> {
    const response = await apiClient.get(`${this.baseUrl}/sessions`);
    return response.data;
  }

  async revokeSession(sessionId: string): Promise<{ message: string }> {
    const response = await apiClient.delete(`${this.baseUrl}/sessions/${sessionId}`);
    return response.data;
  }

  async revokeAllSessions(): Promise<{ message: string }> {
    const response = await apiClient.delete(`${this.baseUrl}/sessions`);
    return response.data;
  }
}

// ==================== 导出 ====================

export const authService = new AuthService();
export default authService;
