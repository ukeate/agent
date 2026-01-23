export const isSameStringArray = (current: string[], next: string[]) => {
  if (current === next) return true
  if (current.length !== next.length) return false
  for (let i = 0; i < current.length; i += 1) {
    if (current[i] !== next[i]) return false
  }
  return true
}
