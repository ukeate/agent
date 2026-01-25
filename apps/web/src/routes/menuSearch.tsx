import type { MenuProps } from 'antd'
import {
  MENU_INDEX,
  MENU_KEY_SET,
  getMenuParentLabelPath,
  getMenuSearchText,
  type MenuItem,
} from './menuIndex'
import { resolveMenuKey, resolveMenuPath } from './menuConfig'
import { ROUTE_PATHS, isKnownRoutePath } from './routeCatalog'
import {
  buildSearchIndexText,
  normalizeSearchText,
  tokenizeSearchQuery,
} from '../utils/searchText'

const scoreMenuMatch = (
  searchText: string,
  tokens: string[],
  normalizedQuery: string
) => {
  if (!tokens.length) return -1
  let score = 0
  for (const token of tokens) {
    const index = searchText.indexOf(token)
    if (index < 0) return -1
    score += index === 0 ? 6 : index < 6 ? 4 : 2
  }
  if (searchText.startsWith(normalizedQuery)) score += 4
  else if (searchText.includes(normalizedQuery)) score += 2
  return score
}

export const filterMenuItems = (
  items: MenuProps['items'],
  query: string
): MenuProps['items'] => {
  if (!items) return items
  const normalizedQuery = normalizeSearchText(query)
  if (!normalizedQuery) return items
  const tokens = tokenizeSearchQuery(normalizedQuery)
  const filtered: MenuProps['items'] = []

  items.forEach(item => {
    if (!item || typeof item !== 'object') return
    if ('type' in item && item.type === 'divider') return

    const combinedText =
      typeof item.key === 'string'
        ? MENU_INDEX.searchTextByKey.get(item.key) ?? getMenuSearchText(item)
        : getMenuSearchText(item)
    const labelMatch = scoreMenuMatch(combinedText, tokens, normalizedQuery) >= 0

    if ('children' in item && item.children) {
      if (labelMatch) {
        filtered.push(item)
        return
      }
      const children = filterMenuItems(
        item.children as MenuProps['items'],
        normalizedQuery
      )
      if (children && children.length > 0) {
        filtered.push({ ...item, children } as MenuItem)
        return
      }
    }

    if (labelMatch) {
      filtered.push(item)
    }
  })

  return filtered
}

export type MenuSearchBoostOptions = {
  favorites?: Set<string>
  recents?: string[]
  favoriteBoost?: number
  recentBoost?: number
}

type RouteSearchEntry = {
  path: string
  label: string
  searchText: string
  index: number
}

const ROUTE_SEARCH_ENTRIES: RouteSearchEntry[] = ROUTE_PATHS.map(
  (path, index) => {
    if (!path || path === '/') return null
    const menuKey = resolveMenuKey(path)
    if (MENU_KEY_SET.has(menuKey)) return null
    return {
      path,
      label: path,
      searchText: buildSearchIndexText(path),
      index,
    }
  }
).filter(Boolean) as RouteSearchEntry[]

const getMenuBoostScore = (
  key: string,
  options?: MenuSearchBoostOptions
) => {
  if (!options) return 0
  let boost = 0
  if (options.favorites?.has(key)) {
    boost += options.favoriteBoost ?? 6
  }
  if (options.recents && options.recents.length > 0) {
    const index = options.recents.indexOf(key)
    if (index >= 0) {
      const base = options.recentBoost ?? 3
      boost += base + Math.max(0, options.recents.length - index - 1)
    }
  }
  return boost
}

export const buildMenuResults = (
  keys: string[],
  query: string,
  limit: number,
  options?: MenuSearchBoostOptions
): MenuItem[] => {
  const normalizedQuery = normalizeSearchText(query)
  if (!normalizedQuery) return []
  const tokens = tokenizeSearchQuery(normalizedQuery)
  const scored = keys
    .map((key, index) => {
      const item = MENU_INDEX.itemByKey.get(key)
      const searchText = MENU_INDEX.searchTextByKey.get(key)
      if (!item || !searchText) return null
      const rawScore = scoreMenuMatch(searchText, tokens, normalizedQuery)
      if (rawScore < 0) return null
      const score = rawScore + getMenuBoostScore(key, options)
      return { item, score, index }
    })
    .filter(Boolean) as Array<{
      item: MenuItem
      score: number
      index: number
    }>
  scored.sort((a, b) => b.score - a.score || a.index - b.index)
  return scored.slice(0, limit).map(entry => entry.item)
}

export const buildNavigationResults = (
  keys: string[],
  query: string,
  limit: number,
  options?: MenuSearchBoostOptions
): MenuItem[] => {
  const normalizedQuery = normalizeSearchText(query)
  if (!normalizedQuery) return []
  const tokens = tokenizeSearchQuery(normalizedQuery)
  if (tokens.length === 0) return []
  const scored: Array<{
    item: MenuItem
    score: number
    index: number
    source: number
  }> = []

  keys.forEach((key, index) => {
    const item = MENU_INDEX.itemByKey.get(key)
    const searchText = MENU_INDEX.searchTextByKey.get(key)
    if (!item || !searchText) return
    const rawScore = scoreMenuMatch(searchText, tokens, normalizedQuery)
    if (rawScore < 0) return
    const score = rawScore + getMenuBoostScore(key, options)
    scored.push({ item, score, index, source: 0 })
  })

  ROUTE_SEARCH_ENTRIES.forEach(entry => {
    const rawScore = scoreMenuMatch(
      entry.searchText,
      tokens,
      normalizedQuery
    )
    if (rawScore < 0) return
    scored.push({
      item: { key: entry.path, label: entry.label } as MenuItem,
      score: rawScore,
      index: entry.index,
      source: 1,
    })
  })

  scored.sort(
    (a, b) => b.score - a.score || a.source - b.source || a.index - b.index
  )
  return scored.slice(0, limit).map(entry => entry.item)
}

export const countMenuMatches = (keys: string[], query: string) => {
  const normalizedQuery = normalizeSearchText(query)
  if (!normalizedQuery) return 0
  const tokens = tokenizeSearchQuery(normalizedQuery)
  if (tokens.length === 0) return 0
  return keys.reduce((count, key) => {
    const searchText = MENU_INDEX.searchTextByKey.get(key)
    if (!searchText) return count
    return scoreMenuMatch(searchText, tokens, normalizedQuery) >= 0
      ? count + 1
      : count
  }, 0)
}

export const countRouteMatches = (query: string) => {
  const normalizedQuery = normalizeSearchText(query)
  if (!normalizedQuery) return 0
  const tokens = tokenizeSearchQuery(normalizedQuery)
  if (tokens.length === 0) return 0
  return ROUTE_SEARCH_ENTRIES.reduce((count, entry) => {
    return scoreMenuMatch(entry.searchText, tokens, normalizedQuery) >= 0
      ? count + 1
      : count
  }, 0)
}

const extractHashRoute = (value: string) => {
  if (!value) return ''
  const hashIndex = value.indexOf('#')
  if (hashIndex < 0) return ''
  const raw = value.slice(hashIndex + 1)
  if (!raw) return ''
  const cleaned = raw.startsWith('!') ? raw.slice(1) : raw
  return cleaned.startsWith('/') ? cleaned : ''
}

const mergeHashRoute = (basePath: string, hashRoute: string) => {
  if (!hashRoute) return basePath
  const trimmedBase =
    basePath && basePath !== '/' ? basePath.replace(/\/+$/, '') : ''
  return `${trimmedBase}${hashRoute}`
}

const ensureHostPath = (value: string) => {
  const separatorIndex = value.search(/[?#]/)
  if (separatorIndex >= 0) {
    return `${value.slice(0, separatorIndex)}/${value.slice(separatorIndex)}`
  }
  return `${value}/`
}

const extractPathFromUrl = (value: string) => {
  if (!value) return ''
  const isProtocolRelative = value.startsWith('//')
  const hasScheme = value.includes('://')
  let candidate = value
  if (!isProtocolRelative && !hasScheme) {
    const hasSlash = value.includes('/')
    const host = hasSlash ? value.split('/')[0] ?? '' : value
    const isLocalhost = host === 'localhost' || host.startsWith('localhost:')
    const hasPort = /:\d+$/.test(host)
    const isIpV4 = /^\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?$/.test(host)
    const looksLikeHost = isLocalhost || hasPort || isIpV4 || host.includes('.')
    if (!looksLikeHost) return ''
    candidate = `http://${hasSlash ? value : ensureHostPath(value)}`
  }
  try {
    const url = new URL(isProtocolRelative ? `http:${value}` : candidate)
    const hashRoute = extractHashRoute(url.hash)
    if (hashRoute) {
      return mergeHashRoute(url.pathname || '/', hashRoute)
    }
    const pathname = url.pathname || '/'
    return `${pathname}${url.search}${url.hash}`
  } catch {
    return ''
  }
}

export const normalizeDirectPath = (value: string) => {
  const trimmed = value.trim()
  if (!trimmed) return ''
  const compact = trimmed.replace(/\s+/g, '')
  const fromUrl = extractPathFromUrl(compact)
  const normalized = fromUrl || compact
  const hashRoute = extractHashRoute(normalized)
  const normalizedPath = hashRoute
    ? mergeHashRoute(normalized.replace(/[?#].*$/, ''), hashRoute)
    : normalized
  const withSlash = normalizedPath.startsWith('/')
    ? normalizedPath
    : normalizedPath.includes('/')
      ? `/${normalizedPath}`
      : ''
  if (!withSlash) return ''
  const separatorIndex = withSlash.search(/[?#]/)
  const pathPart =
    separatorIndex >= 0 ? withSlash.slice(0, separatorIndex) : withSlash
  const suffix = separatorIndex >= 0 ? withSlash.slice(separatorIndex) : ''
  const collapsed = pathPart.replace(/\/{2,}/g, '/')
  const cleaned = collapsed !== '/' ? collapsed.replace(/\/+$/, '') : collapsed
  return cleaned + suffix
}

export type DirectNavigation = {
  path: string
  menuKey: string
}

export type DirectNavigationTarget = {
  path: string
  menuKey: string
  targetPath: string
  isRegistered: boolean
}

export type DirectNavigationMeta = DirectNavigationTarget & {
  known: boolean
  label: string
}

export const resolveDirectNavigation = (
  value: string,
  validKeys: Set<string>
): DirectNavigation => {
  const path = normalizeDirectPath(value)
  if (!path) return { path: '', menuKey: '' }
  const basePath = path.replace(/[?#].*$/, '')
  const menuKey = resolveMenuKey(basePath)
  return { path, menuKey: validKeys.has(menuKey) ? menuKey : '' }
}

export const resolveNavigationPath = (value: string) => {
  if (!value) return ''
  if (value.startsWith('/')) return value
  return resolveMenuPath(value)
}

export const resolveDirectNavigationTarget = (
  value: string,
  validKeys: Set<string>
): DirectNavigationTarget => {
  const { path, menuKey } = resolveDirectNavigation(value, validKeys)
  if (!path) {
    return { path: '', menuKey: '', targetPath: '', isRegistered: false }
  }
  const basePath = path.replace(/[?#].*$/, '')
  const suffix = path.slice(basePath.length)
  if (menuKey) {
    const resolved = resolveMenuPath(menuKey)
    return {
      path,
      menuKey,
      targetPath: `${resolved}${suffix}`,
      isRegistered: true,
    }
  }
  return { path, menuKey: '', targetPath: path, isRegistered: false }
}

const buildDirectNavigationLabel = (
  target: DirectNavigationTarget,
  known: boolean
) => {
  if (!target.path) return ''
  if (target.isRegistered) return `直达 ${target.targetPath}`
  return `直达 ${target.targetPath}${known ? '（未收录）' : '（未注册）'}`
}

export const resolveDirectNavigationMeta = (
  value: string,
  validKeys: Set<string>
): DirectNavigationMeta => {
  const target = resolveDirectNavigationTarget(value, validKeys)
  if (!target.path) {
    return {
      ...target,
      known: false,
      label: '',
    }
  }
  const known = target.isRegistered || isKnownRoutePath(target.targetPath)
  return {
    ...target,
    known,
    label: buildDirectNavigationLabel(target, known),
  }
}

export const getMenuMetaText = (menuKey: string) => {
  const parentPath = getMenuParentLabelPath(menuKey)
  const resolvedPath = resolveNavigationPath(menuKey)
  return parentPath ? `${parentPath} · ${resolvedPath}` : resolvedPath
}

export const getNavigationMetaText = (menuKey: string) => {
  if (MENU_KEY_SET.has(menuKey)) return getMenuMetaText(menuKey)
  if (menuKey.startsWith('/')) return `未收录 · ${menuKey}`
  return getMenuMetaText(menuKey)
}

export const collectMenuKeys = (items: MenuProps['items']) => {
  const keys: string[] = []
  const walk = (list: MenuProps['items']) => {
    if (!list) return
    list.forEach(item => {
      if (!item || typeof item !== 'object') return
      if ('children' in item && item.children) {
        walk(item.children)
        return
      }
      if ('type' in item && item.type === 'group') return
      if (typeof item.key === 'string') keys.push(item.key)
    })
  }
  walk(items)
  return keys
}

export const collectSubmenuKeys = (items: MenuProps['items']) => {
  const keys: string[] = []
  const walk = (list: MenuProps['items']) => {
    if (!list) return
    list.forEach(item => {
      if (!item || typeof item !== 'object') return
      if ('children' in item && item.children) {
        if ('type' in item && item.type === 'group') {
          walk(item.children)
          return
        }
        if (typeof item.key === 'string') keys.push(item.key)
        walk(item.children)
      }
    })
  }
  walk(items)
  return keys
}

export const getMenuItemIcon = (item: MenuItem) => {
  if (!item || typeof item !== 'object') return undefined
  return 'icon' in item ? item.icon : undefined
}
