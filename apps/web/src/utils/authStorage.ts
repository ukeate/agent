const AUTH_TOKEN_KEY = 'auth_token'
const REFRESH_TOKEN_KEY = 'refresh_token'

const hasStorage = () =>
  typeof window !== 'undefined' && typeof window.localStorage !== 'undefined'

export const getStoredAuthToken = (): string | null => {
  if (!hasStorage()) return null
  return window.localStorage.getItem(AUTH_TOKEN_KEY)
}

export const getStoredRefreshToken = (): string | null => {
  if (!hasStorage()) return null
  return window.localStorage.getItem(REFRESH_TOKEN_KEY)
}

export const setStoredTokens = (
  accessToken: string,
  refreshToken?: string
): void => {
  if (!hasStorage()) return
  window.localStorage.setItem(AUTH_TOKEN_KEY, accessToken)
  if (refreshToken) {
    window.localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken)
  }
}

export const clearStoredTokens = (): void => {
  if (!hasStorage()) return
  window.localStorage.removeItem(AUTH_TOKEN_KEY)
  window.localStorage.removeItem(REFRESH_TOKEN_KEY)
}
