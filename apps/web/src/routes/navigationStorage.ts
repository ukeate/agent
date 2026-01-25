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

const getStorage = () => {
  if (typeof window === 'undefined') return null
  try {
    return window.localStorage
  } catch {
    return null
  }
}

const readStorageItem = (key: string) => {
  const storage = getStorage()
  if (!storage) return null
  try {
    return storage.getItem(key)
  } catch {
    return null
  }
}

const writeStorageItem = (key: string, value: string) => {
  const storage = getStorage()
  if (!storage) return
  try {
    storage.setItem(key, value)
  } catch {
    return
  }
}

const dispatchMenuStorageEvent = (eventName: string, detail?: unknown) => {
  if (typeof window === 'undefined') return
  window.dispatchEvent(new CustomEvent(eventName, { detail }))
}

export const readStoredMenuCollapsed = (): boolean => {
  return readStorageItem(MENU_COLLAPSED_STORAGE_KEY) === '1'
}

export const writeStoredMenuCollapsed = (collapsed: boolean) => {
  writeStorageItem(MENU_COLLAPSED_STORAGE_KEY, collapsed ? '1' : '0')
}

export const readStoredMenuRecents = (validKeys: Set<string>): string[] => {
  const stored = readStorageItem(MENU_RECENTS_STORAGE_KEY)
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
  writeStorageItem(MENU_RECENTS_STORAGE_KEY, JSON.stringify(keys))
  dispatchMenuStorageEvent(MENU_STORAGE_EVENTS.recents, keys)
}

export const readStoredMenuFavorites = (validKeys: Set<string>): string[] => {
  const stored = readStorageItem(MENU_FAVORITES_STORAGE_KEY)
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
  writeStorageItem(MENU_FAVORITES_STORAGE_KEY, JSON.stringify(keys))
  dispatchMenuStorageEvent(MENU_STORAGE_EVENTS.favorites, keys)
}

export const readStoredMenuOpenKeys = (
  validKeys: Set<string>
): string[] => {
  const stored = readStorageItem(MENU_OPEN_KEYS_STORAGE_KEY)
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
  writeStorageItem(MENU_OPEN_KEYS_STORAGE_KEY, JSON.stringify(keys))
}

export const readStoredLastRoute = (
  validKeys: Set<string>
): string | null => {
  const stored = readStorageItem(MENU_LAST_ROUTE_STORAGE_KEY)
  if (!stored) return null
  const normalized = stored.trim()
  if (!normalized) return null
  const withSlash = normalized.startsWith('/') ? normalized : `/${normalized}`
  const basePath = withSlash.replace(/[?#].*$/, '')
  const key = resolveMenuKey(basePath)
  if (!validKeys.has(key)) return null
  const resolvedBase = resolveMenuPath(key)
  const suffix = withSlash.slice(basePath.length)
  return `${resolvedBase}${suffix}`
}

export const writeStoredLastRoute = (
  path: string,
  validKeys: Set<string>
) => {
  const normalized = path.trim()
  if (!normalized) return
  const withSlash = normalized.startsWith('/') ? normalized : `/${normalized}`
  const basePath = withSlash.replace(/[?#].*$/, '')
  const key = resolveMenuKey(basePath)
  if (!validKeys.has(key)) return
  const resolvedBase = resolveMenuPath(key)
  const suffix = withSlash.slice(basePath.length)
  const nextPath = `${resolvedBase}${suffix}`
  writeStorageItem(MENU_LAST_ROUTE_STORAGE_KEY, nextPath)
  dispatchMenuStorageEvent(MENU_STORAGE_EVENTS.lastRoute, nextPath)
}
