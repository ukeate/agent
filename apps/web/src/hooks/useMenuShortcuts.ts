import { useCallback, useEffect, useState } from 'react'
import { isSameStringArray } from '../utils/array'
import {
  MENU_STORAGE_EVENTS,
  readStoredMenuFavorites,
  readStoredMenuRecents,
  writeStoredMenuFavorites,
  writeStoredMenuRecents,
} from '../routes/navigationStorage'

type KeysUpdater = string[] | ((prev: string[]) => string[])

export const useMenuShortcuts = (validKeys: Set<string>) => {
  const [favoriteKeys, setFavoriteState] = useState<string[]>(() =>
    readStoredMenuFavorites(validKeys)
  )
  const [recentKeys, setRecentState] = useState<string[]>(() =>
    readStoredMenuRecents(validKeys)
  )

  const filterValidKeys = useCallback(
    (keys: string[]) => keys.filter(key => validKeys.has(key)),
    [validKeys]
  )

  const setFavoriteKeys = useCallback(
    (updater: KeysUpdater) => {
      setFavoriteState(prev => {
        const next = typeof updater === 'function' ? updater(prev) : updater
        const filtered = filterValidKeys(next)
        writeStoredMenuFavorites(filtered)
        return filtered
      })
    },
    [filterValidKeys]
  )

  const setRecentKeys = useCallback(
    (updater: KeysUpdater) => {
      setRecentState(prev => {
        const next = typeof updater === 'function' ? updater(prev) : updater
        const filtered = filterValidKeys(next)
        writeStoredMenuRecents(filtered)
        return filtered
      })
    },
    [filterValidKeys]
  )

  const syncFromStorage = useCallback(() => {
    const nextFavorites = readStoredMenuFavorites(validKeys)
    const nextRecents = readStoredMenuRecents(validKeys)
    setFavoriteState(prev =>
      isSameStringArray(prev, nextFavorites) ? prev : nextFavorites
    )
    setRecentState(prev =>
      isSameStringArray(prev, nextRecents) ? prev : nextRecents
    )
  }, [validKeys])

  useEffect(() => {
    if (typeof window === 'undefined') return
    syncFromStorage()
    const handleStorage = () => syncFromStorage()
    window.addEventListener(MENU_STORAGE_EVENTS.favorites, handleStorage)
    window.addEventListener(MENU_STORAGE_EVENTS.recents, handleStorage)
    window.addEventListener('storage', handleStorage)
    return () => {
      window.removeEventListener(MENU_STORAGE_EVENTS.favorites, handleStorage)
      window.removeEventListener(MENU_STORAGE_EVENTS.recents, handleStorage)
      window.removeEventListener('storage', handleStorage)
    }
  }, [syncFromStorage])

  return {
    favoriteKeys,
    recentKeys,
    setFavoriteKeys,
    setRecentKeys,
  }
}

export default useMenuShortcuts
