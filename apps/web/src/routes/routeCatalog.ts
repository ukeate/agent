import rawRoutePaths from '../route_paths.txt?raw'

const normalizeRoutePath = (value: string) => {
  const trimmed = value.trim()
  if (!trimmed) return ''
  const withSlash = trimmed.startsWith('/') ? trimmed : `/${trimmed}`
  const base = withSlash.replace(/[?#].*$/, '')
  if (base === '/') return base
  return base.replace(/\/+$/, '')
}

const parseRoutePaths = (raw: string) => {
  const paths: string[] = []
  raw
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean)
    .forEach(line => {
      const match = line.match(/path="([^"]+)"/)
      const value = match?.[1] ?? line
      const normalized = normalizeRoutePath(value)
      if (normalized) paths.push(normalized)
    })
  return Array.from(new Set(paths))
}

export const ROUTE_PATHS = parseRoutePaths(rawRoutePaths)
export const ROUTE_PATH_SET = new Set(ROUTE_PATHS)

export const isKnownRoutePath = (value: string) => {
  const normalized = normalizeRoutePath(value)
  if (!normalized) return false
  return ROUTE_PATH_SET.has(normalized)
}

export default ROUTE_PATHS
