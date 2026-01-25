export const normalizeSearchText = (value: string): string => {
  return value
    .toLowerCase()
    .replace(/[\\/_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

const splitSearchText = (value: string): string => {
  return value
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
    .replace(/([a-zA-Z])(\d)/g, '$1 $2')
    .replace(/(\d)([a-zA-Z])/g, '$1 $2')
}

export const buildSearchIndexText = (value: string): string => {
  const normalized = normalizeSearchText(value)
  if (!normalized) return ''
  const spaced = normalizeSearchText(splitSearchText(value))
  const compact = normalized.replace(/\s+/g, '')
  const acronym = normalized
    .split(' ')
    .filter(Boolean)
    .map(token => token[0])
    .join('')
  return Array.from(
    new Set([normalized, spaced, compact, acronym].filter(Boolean))
  ).join(' ')
}

export const splitSearchTokens = (value: string): string[] => {
  return tokenizeSearchQuery(value)
}

export const resolveDeferredQuery = (
  value: string,
  deferredValue: string
): string => {
  if (!value.trim()) return ''
  return deferredValue.trim() ? deferredValue : value
}

export const matchSearchTokens = (
  searchText: string,
  tokens: string[]
): boolean => {
  if (tokens.length === 0) return true
  return tokens.every(token => searchText.includes(token))
}

export const tokenizeSearchQuery = (value: string): string[] => {
  const normalized = normalizeSearchText(value)
  if (!normalized) return []
  const tokens: string[] = []
  const pattern = /"([^"]+)"|([^\s]+)/g
  let match: RegExpExecArray | null
  while ((match = pattern.exec(normalized))) {
    const token = (match[1] ?? match[2] ?? '').trim()
    if (token) tokens.push(token)
  }
  return tokens
}
