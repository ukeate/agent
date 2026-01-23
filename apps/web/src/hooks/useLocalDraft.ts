import { useEffect, useRef, useState } from 'react'
import { logger } from '../utils/logger'

const hasStorage = () =>
  typeof window !== 'undefined' && typeof window.localStorage !== 'undefined'

export const useLocalDraft = (storageKey: string, initialValue = '') => {
  const [value, setValue] = useState(initialValue)
  const skipPersistRef = useRef(true)

  useEffect(() => {
    if (!hasStorage()) return
    skipPersistRef.current = true
    try {
      const stored = window.localStorage.getItem(storageKey)
      setValue(stored ?? initialValue)
    } catch (error) {
      logger.warn('读取草稿失败', error)
      setValue(initialValue)
    }
  }, [storageKey, initialValue])

  useEffect(() => {
    if (!hasStorage()) return
    if (skipPersistRef.current) {
      skipPersistRef.current = false
      return
    }
    try {
      if (value) {
        window.localStorage.setItem(storageKey, value)
      } else {
        window.localStorage.removeItem(storageKey)
      }
    } catch (error) {
      logger.warn('保存草稿失败', error)
    }
  }, [storageKey, value])

  return [value, setValue] as const
}
