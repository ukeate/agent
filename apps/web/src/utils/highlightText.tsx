import { splitSearchTokens } from './searchText'

const escapeRegExp = (value: string) => {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

export const renderHighlightedText = (text: string, query: string) => {
  if (!text) return text
  const tokens = splitSearchTokens(query)
  if (tokens.length === 0) return text
  const uniqueTokens = Array.from(new Set(tokens))
  const pattern = uniqueTokens.map(escapeRegExp).join('|')
  if (!pattern) return text
  const regex = new RegExp(`(${pattern})`, 'ig')
  const parts = text.split(regex)
  return parts.map((part, index) =>
    index % 2 === 1 ? (
      <mark
        key={`${part}-${index}`}
        style={{
          backgroundColor: '#fef3c7',
          color: '#b45309',
          borderRadius: 4,
          padding: '0 2px',
        }}
      >
        {part}
      </mark>
    ) : (
      <span key={`${part}-${index}`}>{part}</span>
    )
  )
}

export default renderHighlightedText
