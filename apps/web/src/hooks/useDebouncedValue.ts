import { useEffect, useState } from 'react'

export const useDebouncedValue = <T>(value: T, delayMs = 300): T => {
  const [debounced, setDebounced] = useState(value)

  useEffect(() => {
    if (delayMs <= 0) {
      setDebounced(value)
      return
    }
    const timer = setTimeout(() => setDebounced(value), delayMs)
    return () => clearTimeout(timer)
  }, [delayMs, value])

  return debounced
}
