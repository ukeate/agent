import React from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize'
import { splitSearchTokens } from '../../utils/searchText'

type HastNode = {
  type: string
  tagName?: string
  value?: string
  children?: HastNode[]
}

const SANITIZE_SCHEMA = {
  ...defaultSchema,
  tagNames: Array.from(
    new Set([...(defaultSchema.tagNames ?? []), 'mark'])
  ),
}

const HIGHLIGHT_SKIP_TAGS = new Set(['code', 'pre'])

const escapeRegExp = (value: string) => {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

const buildHighlightRegex = (query?: string) => {
  if (!query) return null
  const tokens = splitSearchTokens(query)
  if (tokens.length === 0) return null
  const pattern = Array.from(new Set(tokens))
    .map(escapeRegExp)
    .filter(Boolean)
    .join('|')
  if (!pattern) return null
  return new RegExp(`(${pattern})`, 'ig')
}

const buildHighlightNodes = (value: string, regex: RegExp): HastNode[] => {
  const parts = value.split(regex)
  if (parts.length <= 1) return [{ type: 'text', value }]
  const nodes: HastNode[] = []
  parts.forEach((part, index) => {
    if (!part) return
    if (index % 2 === 1) {
      nodes.push({
        type: 'element',
        tagName: 'mark',
        children: [{ type: 'text', value: part }],
      })
      return
    }
    nodes.push({ type: 'text', value: part })
  })
  return nodes
}

const applyHighlight = (
  node: HastNode,
  regex: RegExp,
  parents: string[] = []
) => {
  if (!node || typeof node !== 'object') return
  const children = node.children
  if (!children || !Array.isArray(children)) return
  const tagName = node.type === 'element' ? node.tagName : undefined
  const nextParents = tagName ? [...parents, tagName] : parents
  if (nextParents.some(tag => HIGHLIGHT_SKIP_TAGS.has(tag))) return
  const nextChildren: HastNode[] = []
  children.forEach(child => {
    if (child.type === 'text' && typeof child.value === 'string') {
      nextChildren.push(...buildHighlightNodes(child.value, regex))
      return
    }
    applyHighlight(child, regex, nextParents)
    nextChildren.push(child)
  })
  node.children = nextChildren
}

const createHighlightPlugin = (query?: string) => {
  const regex = buildHighlightRegex(query)
  if (!regex) return null
  return () => (tree: HastNode) => {
    applyHighlight(tree, regex)
  }
}

interface MarkdownRendererProps {
  content: string
  className?: string
  highlightQuery?: string
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({
  content,
  className = '',
  highlightQuery,
}) => {
  const highlightPlugin = createHighlightPlugin(highlightQuery)
  const rehypePlugins = highlightPlugin
    ? [rehypeHighlight, highlightPlugin, [rehypeSanitize, SANITIZE_SCHEMA]]
    : [rehypeHighlight, [rehypeSanitize, SANITIZE_SCHEMA]]
  return (
    <div className={`markdown-content ${className}`}>
      <ReactMarkdown
        rehypePlugins={rehypePlugins}
        components={{
          // 自定义代码块样式 - 处理多行代码
          code: ({ node, className, children, ...props }) => {
            const isInline = !className?.includes('language-')

            if (!isInline) {
              // 保持原始内容，不移除换行符，确保代码格式正确
              return (
                <pre
                  style={{
                    whiteSpace: 'pre',
                    overflowX: 'auto',
                    fontFamily: 'monospace',
                    lineHeight: '1.5',
                    padding: '1rem',
                    backgroundColor: '#f8f9fa',
                    borderRadius: '0.375rem',
                    border: '1px solid #e9ecef',
                  }}
                >
                  <code
                    className={className}
                    {...props}
                    style={{
                      whiteSpace: 'pre',
                      fontFamily: 'inherit',
                      fontSize: '0.875rem',
                    }}
                  >
                    {children}
                  </code>
                </pre>
              )
            }

            return (
              <code className="inline-code" {...props}>
                {children}
              </code>
            )
          },
          // 保留段落的换行
          p: ({ children }) => (
            <p style={{ whiteSpace: 'pre-wrap' }}>{children}</p>
          ),
          // 自定义预格式化文本
          pre: ({ children }) => (
            <pre style={{ whiteSpace: 'pre', overflowX: 'auto' }}>
              {children}
            </pre>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}

export default MarkdownRenderer
export { MarkdownRenderer }
