const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

const API_PREFIX = '/api/v1';

const ABSOLUTE_URL = /^https?:\/\//i;

const resolveApiUrl = (endpoint: string): string => {
  if (!endpoint) return `${API_BASE_URL}${API_PREFIX}`;
  if (ABSOLUTE_URL.test(endpoint)) return endpoint;
  const normalized = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  if (normalized.startsWith(API_PREFIX)) return `${API_BASE_URL}${normalized}`;
  return `${API_BASE_URL}${API_PREFIX}${normalized}`;
};

export const buildApiUrl = (endpoint: string): string => {
  return resolveApiUrl(endpoint);
};

const extractErrorMessage = async (response: Response): Promise<string> => {
  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    try {
      const data = await response.json();
      return (
        data?.detail ||
        data?.error ||
        data?.message ||
        JSON.stringify(data)
      );
    } catch {
      return `${response.status} ${response.statusText}`.trim();
    }
  }
  try {
    const text = await response.text();
    if (text) return text;
  } catch {
    return `${response.status} ${response.statusText}`.trim();
  }
  return `${response.status} ${response.statusText}`.trim();
};

export const apiFetch = async (endpoint: string, init?: RequestInit): Promise<Response> => {
  const response = await fetch(resolveApiUrl(endpoint), init);
  if (!response.ok) {
    const message = await extractErrorMessage(response);
    throw new Error(message);
  }
  return response;
};

export const apiFetchJson = async <T>(endpoint: string, init?: RequestInit): Promise<T> => {
  const response = await apiFetch(endpoint, init);
  if (response.status === 204) return null as T;
  return response.json() as Promise<T>;
};

export const buildWsUrl = (endpoint: string): string => {
  const httpUrl = buildApiUrl(endpoint);
  if (httpUrl.startsWith('https://')) {
    return `wss://${httpUrl.slice('https://'.length)}`;
  }
  if (httpUrl.startsWith('http://')) {
    return `ws://${httpUrl.slice('http://'.length)}`;
  }
  if (httpUrl.startsWith('/')) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}${httpUrl}`;
  }
  return httpUrl;
};
