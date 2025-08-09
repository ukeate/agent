import React from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import rehypeSanitize from 'rehype-sanitize'

interface MarkdownRendererProps {
  content: string
  className?: string
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ 
  content, 
  className = '' 
}) => {
  return (
    <div className={`markdown-content ${className}`}>
      <ReactMarkdown
        rehypePlugins={[rehypeHighlight, rehypeSanitize]}
        components={{
          // 自定义代码块样式 - 处理多行代码
          code: ({ node, className, children, ...props }) => {
            const isInline = !className?.includes('language-')
            
            if (!isInline) {
              // 保持原始内容，不移除换行符，确保代码格式正确
              return (
                <pre style={{ 
                  whiteSpace: 'pre', 
                  overflowX: 'auto',
                  fontFamily: 'monospace',
                  lineHeight: '1.5',
                  padding: '1rem',
                  backgroundColor: '#f8f9fa',
                  borderRadius: '0.375rem',
                  border: '1px solid #e9ecef'
                }}>
                  <code className={className} {...props} style={{
                    whiteSpace: 'pre',
                    fontFamily: 'inherit',
                    fontSize: '0.875rem'
                  }}>
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