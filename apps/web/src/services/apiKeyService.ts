import apiClient from './apiClient';

export interface APIKey {
  id: string;
  name: string;
  key: string;
  created_at: string;
  expires_at: string | null;
  permissions: string[];
  status: 'active' | 'expired' | 'revoked';
  description?: string;
}

export interface CreateAPIKeyRequest {
  name: string;
  description?: string;
  expires_in_days?: number;
  permissions: string[];
}

class ApiKeyService {
  private baseUrl = '/security/api-keys';

  async listApiKeys(): Promise<APIKey[]> {
    const response = await apiClient.get<{ api_keys: APIKey[] }>(this.baseUrl);
    return response.data.api_keys || [];
  }

  async createApiKey(data: CreateAPIKeyRequest): Promise<APIKey> {
    const response = await apiClient.post(this.baseUrl, data);
    return response.data;
  }

  async revokeApiKey(keyId: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/${keyId}`);
  }

  async getApiKeyPermissions(): Promise<string[]> {
    const response = await apiClient.get<{ permissions: string[] }>(`${this.baseUrl}/permissions`);
    return response.data.permissions || [];
  }
}

export const apiKeyService = new ApiKeyService();
