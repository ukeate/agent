import type { MenuProps } from 'antd'
import {
  MENU_INDEX,
  getMenuParentLabelPath,
  getMenuSearchText,
  type MenuItem,
} from './menuIndex'
import { resolveMenuKey, resolveMenuPath } from './menuConfig'
import { normalizeSearchText } from '../utils/searchText'

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
  const tokens = normalizedQuery.split(' ').filter(Boolean)
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
  const tokens = normalizedQuery.split(' ').filter(Boolean)
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

export const normalizeDirectPath = (value: string) => {
  const trimmed = value.trim()
  if (!trimmed) return ''
  const normalized = trimmed.replace(/\s+/g, '')
  const withSlash = normalized.startsWith('/')
    ? normalized
    : normalized.includes('/')
      ? `/${normalized}`
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

export const getMenuMetaText = (menuKey: string) => {
  const parentPath = getMenuParentLabelPath(menuKey)
  const resolvedPath = resolveNavigationPath(menuKey)
  return parentPath ? `${parentPath} · ${resolvedPath}` : resolvedPath
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
