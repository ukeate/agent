import { resolveMenuKey, resolveMenuPath } from './menuConfig'

const MENU_COLLAPSED_STORAGE_KEY = 'ai-agent-menu-collapsed'
const MENU_FAVORITES_STORAGE_KEY = 'ai-agent-menu-favorites'
const MENU_RECENTS_STORAGE_KEY = 'ai-agent-menu-recents'
const MENU_OPEN_KEYS_STORAGE_KEY = 'ai-agent-menu-open-keys'
const MENU_LAST_ROUTE_STORAGE_KEY = 'ai-agent-last-route'

export const MENU_STORAGE_EVENTS = {
  favorites: 'ai-agent-menu-favorites-updated',
  recents: 'ai-agent-menu-recents-updated',
  lastRoute: 'ai-agent-last-route-updated',
} as const

const hasStorage = () =>
  typeof window !== 'undefined' && typeof window.localStorage !== 'undefined'

const dispatchMenuStorageEvent = (eventName: string, detail?: unknown) => {
  if (typeof window === 'undefined') return
  window.dispatchEvent(new CustomEvent(eventName, { detail }))
}

export const readStoredMenuCollapsed = (): boolean => {
  if (!hasStorage()) return false
  return window.localStorage.getItem(MENU_COLLAPSED_STORAGE_KEY) === '1'
}

export const writeStoredMenuCollapsed = (collapsed: boolean) => {
  if (!hasStorage()) return
  window.localStorage.setItem(
    MENU_COLLAPSED_STORAGE_KEY,
    collapsed ? '1' : '0'
  )
}

export const readStoredMenuRecents = (validKeys: Set<string>): string[] => {
  if (!hasStorage()) return []
  const stored = window.localStorage.getItem(MENU_RECENTS_STORAGE_KEY)
  if (!stored) return []
  try {
    const parsed = JSON.parse(stored)
    if (!Array.isArray(parsed)) return []
    const filtered = parsed.filter(
      item => typeof item === 'string' && validKeys.has(item)
    )
    return Array.from(new Set(filtered))
  } catch {
    return []
  }
}

export const writeStoredMenuRecents = (keys: string[]) => {
  if (!hasStorage()) return
  window.localStorage.setItem(MENU_RECENTS_STORAGE_KEY, JSON.stringify(keys))
  dispatchMenuStorageEvent(MENU_STORAGE_EVENTS.recents, keys)
}

export const readStoredMenuFavorites = (validKeys: Set<string>): string[] => {
  if (!hasStorage()) return []
  const stored = window.localStorage.getItem(MENU_FAVORITES_STORAGE_KEY)
  if (!stored) return []
  try {
    const parsed = JSON.parse(stored)
    if (!Array.isArray(parsed)) return []
    const filtered = parsed.filter(
      item => typeof item === 'string' && validKeys.has(item)
    )
    return Array.from(new Set(filtered))
  } catch {
    return []
  }
}

export const writeStoredMenuFavorites = (keys: string[]) => {
  if (!hasStorage()) return
  window.localStorage.setItem(MENU_FAVORITES_STORAGE_KEY, JSON.stringify(keys))
  dispatchMenuStorageEvent(MENU_STORAGE_EVENTS.favorites, keys)
}

export const readStoredMenuOpenKeys = (
  validKeys: Set<string>
): string[] => {
  if (!hasStorage()) return []
  const stored = window.localStorage.getItem(MENU_OPEN_KEYS_STORAGE_KEY)
  if (!stored) return []
  try {
    const parsed = JSON.parse(stored)
    if (!Array.isArray(parsed)) return []
    return parsed.filter(
      key => typeof key === 'string' && validKeys.has(key)
    )
  } catch {
    return []
  }
}

export const writeStoredMenuOpenKeys = (keys: string[]) => {
  if (!hasStorage()) return
  window.localStorage.setItem(MENU_OPEN_KEYS_STORAGE_KEY, JSON.stringify(keys))
}

export const readStoredLastRoute = (
  validKeys: Set<string>
): string | null => {
  if (!hasStorage()) return null
  const stored = window.localStorage.getItem(MENU_LAST_ROUTE_STORAGE_KEY)
  if (!stored) return null
  const normalized = stored.startsWith('/') ? stored : `/${stored}`
  const key = resolveMenuKey(normalized)
  if (!validKeys.has(key)) return null
  return resolveMenuPath(key)
}

export const writeStoredLastRoute = (
  path: string,
  validKeys: Set<string>
) => {
  if (!hasStorage()) return
  const normalized = path.startsWith('/') ? path : `/${path}`
  const key = resolveMenuKey(normalized)
  if (!validKeys.has(key)) return
  const nextPath = resolveMenuPath(key)
  window.localStorage.setItem(MENU_LAST_ROUTE_STORAGE_KEY, nextPath)
  dispatchMenuStorageEvent(MENU_STORAGE_EVENTS.lastRoute, nextPath)
}
