// import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import MarkdownRenderer from '../../src/components/ui/MarkdownRenderer'

describe('MarkdownRenderer', () => {
  it('应该正确渲染基本文本', () => {
    render(<MarkdownRenderer content="Hello World" />)
    expect(screen.getByText('Hello World')).toBeInTheDocument()
  })

  it('应该正确渲染代码块', () => {
    const markdownWithCode = '```javascript\nconst hello = "world";\nconsole.log(hello);\n```'
    render(<MarkdownRenderer content={markdownWithCode} />)
    
    // 语法高亮会将代码分解成多个元素，检查关键词是否存在
    expect(screen.getByText('const')).toBeInTheDocument()
    expect(screen.getByText('"world"')).toBeInTheDocument()
    expect(screen.getByText('console')).toBeInTheDocument()
    expect(screen.getByText('log')).toBeInTheDocument()
  })

  it('应该正确渲染内联代码', () => {
    const markdownWithInlineCode = 'This is `inline code` in text.'
    render(<MarkdownRenderer content={markdownWithInlineCode} />)
    
    expect(screen.getByText('inline code')).toBeInTheDocument()
    // 文本被分解成多个元素，使用部分匹配
    expect(screen.getByText(/This is/)).toBeInTheDocument()
    expect(screen.getByText(/in text\./)).toBeInTheDocument()
  })

  it('应该正确渲染标题', () => {
    const markdownWithHeaders = '# Header 1\n## Header 2\n### Header 3'
    render(<MarkdownRenderer content={markdownWithHeaders} />)
    
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Header 1')
    expect(screen.getByRole('heading', { level: 2 })).toHaveTextContent('Header 2')
    expect(screen.getByRole('heading', { level: 3 })).toHaveTextContent('Header 3')
  })

  it('应该正确渲染列表', () => {
    const markdownWithList = '- Item 1\n- Item 2\n- Item 3'
    render(<MarkdownRenderer content={markdownWithList} />)
    
    expect(screen.getByText('Item 1')).toBeInTheDocument()
    expect(screen.getByText('Item 2')).toBeInTheDocument()
    expect(screen.getByText('Item 3')).toBeInTheDocument()
  })

  it('应该正确渲染引用块', () => {
    const markdownWithBlockquote = '> This is a blockquote'
    render(<MarkdownRenderer content={markdownWithBlockquote} />)
    
    const blockquote = screen.getByText('This is a blockquote').closest('blockquote')
    expect(blockquote).toBeInTheDocument()
  })

  it('应该应用自定义className', () => {
    const { container } = render(
      <MarkdownRenderer content="Test" className="custom-class" />
    )
    
    expect(container.firstChild).toHaveClass('markdown-content', 'custom-class')
  })
})