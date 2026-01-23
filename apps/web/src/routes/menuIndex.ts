import type { MenuProps } from 'antd'
import { MENU_ITEMS, getMenuLabelText, resolveMenuPath } from './menuConfig'
import { buildSearchIndexText } from '../utils/searchText'

type RawMenuItem = NonNullable<MenuProps['items']>[number]
export type MenuItem = Exclude<RawMenuItem, { type: 'divider' }>

export type MenuIndex = {
  itemByKey: Map<string, MenuItem>
  searchTextByKey: Map<string, string>
  parentKeysByKey: Map<string, string[]>
  menuKeys: string[]
  submenuKeys: string[]
}

export const getMenuSearchText = (
  item: MenuItem,
  parentLabels: string[] = []
) => {
  const labelText = getMenuLabelText(item.label)
  const keyText = typeof item.key === 'string' ? item.key : ''
  const pathText = typeof item.key === 'string' ? resolveMenuPath(item.key) : ''
  const parentText = parentLabels.length ? parentLabels.join(' ') : ''
  return buildSearchIndexText(
    `${labelText} ${parentText} ${keyText} ${pathText}`
  )
}

const buildMenuIndex = (items: MenuProps['items']): MenuIndex => {
  const itemByKey = new Map<string, MenuItem>()
  const searchTextByKey = new Map<string, string>()
  const parentKeysByKey = new Map<string, string[]>()
  const menuKeys: string[] = []
  const submenuKeys: string[] = []

  const walk = (
    list: MenuProps['items'],
    parents: string[],
    parentLabels: string[]
  ) => {
    if (!list) return
    list.forEach(item => {
      if (!item || typeof item !== 'object') return
      if ('type' in item && item.type === 'divider') return

      const isGroup = 'type' in item && item.type === 'group'
      const key = typeof item.key === 'string' ? item.key : null
      const children = 'children' in item ? item.children : null
      const labelText = getMenuLabelText(item.label)
      const nextParentLabels = labelText
        ? [...parentLabels, labelText]
        : parentLabels

      if (key) {
        itemByKey.set(key, item as MenuItem)
        searchTextByKey.set(
          key,
          getMenuSearchText(item as MenuItem, parentLabels)
        )
        parentKeysByKey.set(key, parents)
      }

      if (children && Array.isArray(children) && children.length > 0) {
        if (key && !isGroup) submenuKeys.push(key)
        const nextParents =
          key && !isGroup ? [...parents, key] : parents
        walk(children as MenuProps['items'], nextParents, nextParentLabels)
        return
      }

      if (key && !isGroup) menuKeys.push(key)
    })
  }

  walk(items, [], [])
  return { itemByKey, searchTextByKey, parentKeysByKey, menuKeys, submenuKeys }
}

export const MENU_INDEX = buildMenuIndex(MENU_ITEMS)
export const MENU_KEY_SET = new Set(MENU_INDEX.menuKeys)
export const MENU_SUBMENU_KEY_SET = new Set(MENU_INDEX.submenuKeys)

export const getMenuParentLabelPath = (menuKey: string): string => {
  const parentKeys = MENU_INDEX.parentKeysByKey.get(menuKey)
  if (!parentKeys || parentKeys.length === 0) return ''
  const labels: string[] = []
  parentKeys.forEach(key => {
    const item = MENU_INDEX.itemByKey.get(key)
    if (!item) return
    const label = getMenuLabelText(item.label)
    if (label) labels.push(label)
  })
  return labels.join(' / ')
}
