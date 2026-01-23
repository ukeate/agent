import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import {
  AgentAvatar,
  RoleBadge,
  StatusIndicator,
} from '../../../src/components/multi-agent/AgentAvatar'
import { Agent } from '../../../src/stores/multiAgentStore'

describe('AgentAvatar', () => {
  const mockAgent: Agent = {
    id: 'test-agent-1',
    name: '测试代码专家',
    role: 'code_expert',
    status: 'active',
    capabilities: ['代码生成', '代码审查', '性能优化'],
    configuration: {
      model: 'gpt-4o-mini',
      temperature: 0.1,
      max_tokens: 2000,
      tools: ['code_analyzer'],
      system_prompt: '你是一位专业的软件开发专家',
    },
    created_at: '2025-01-01T00:00:00Z',
    updated_at: '2025-01-01T00:00:00Z',
  }

  it('应该渲染代码专家头像', () => {
    render(<AgentAvatar agent={mockAgent} />)

    // 检查头像容器
    const avatar = screen.getByTitle('测试代码专家 (code_expert)')
    expect(avatar).toBeInTheDocument()

    // 检查代码专家图标
    expect(avatar).toHaveTextContent('🔧')

    // 检查背景颜色类
    expect(avatar).toHaveClass('bg-blue-500')
  })

  it('应该渲染架构师头像', () => {
    const architectAgent: Agent = {
      ...mockAgent,
      role: 'architect',
      name: '测试架构师',
    }

    render(<AgentAvatar agent={architectAgent} />)

    const avatar = screen.getByTitle('测试架构师 (architect)')
    expect(avatar).toHaveTextContent('🏗️')
    expect(avatar).toHaveClass('bg-green-500')
  })

  it('应该渲染文档专家头像', () => {
    const docAgent: Agent = {
      ...mockAgent,
      role: 'doc_expert',
      name: '测试文档专家',
    }

    render(<AgentAvatar agent={docAgent} />)

    const avatar = screen.getByTitle('测试文档专家 (doc_expert)')
    expect(avatar).toHaveTextContent('📝')
    expect(avatar).toHaveClass('bg-orange-500')
  })

  it('应该渲染监督者头像', () => {
    const supervisorAgent: Agent = {
      ...mockAgent,
      role: 'supervisor',
      name: '测试监督者',
    }

    render(<AgentAvatar agent={supervisorAgent} />)

    const avatar = screen.getByTitle('测试监督者 (supervisor)')
    expect(avatar).toHaveTextContent('👨‍💼')
    expect(avatar).toHaveClass('bg-purple-500')
  })

  it('应该支持不同尺寸', () => {
    const { rerender } = render(<AgentAvatar agent={mockAgent} size="sm" />)
    let avatar = screen.getByTitle('测试代码专家 (code_expert)')
    expect(avatar).toHaveClass('w-8', 'h-8', 'text-sm')

    rerender(<AgentAvatar agent={mockAgent} size="md" />)
    avatar = screen.getByTitle('测试代码专家 (code_expert)')
    expect(avatar).toHaveClass('w-12', 'h-12', 'text-lg')

    rerender(<AgentAvatar agent={mockAgent} size="lg" />)
    avatar = screen.getByTitle('测试代码专家 (code_expert)')
    expect(avatar).toHaveClass('w-16', 'h-16', 'text-xl')
  })

  it('应该显示状态指示器', () => {
    render(<AgentAvatar agent={mockAgent} showStatus={true} />)

    // 检查状态指示器
    const statusIndicator = screen.getByTitle('状态: active')
    expect(statusIndicator).toBeInTheDocument()
    expect(statusIndicator).toHaveClass('bg-green-400')
  })

  it('应该隐藏状态指示器', () => {
    render(<AgentAvatar agent={mockAgent} showStatus={false} />)

    // 状态指示器不应该存在
    const statusIndicator = screen.queryByTitle('状态: active')
    expect(statusIndicator).not.toBeInTheDocument()
  })

  it('应该显示不同状态的指示器', () => {
    const agents = [
      { ...mockAgent, status: 'idle' as const },
      { ...mockAgent, status: 'busy' as const },
      { ...mockAgent, status: 'offline' as const },
    ]

    const { rerender } = render(<AgentAvatar agent={agents[0]} />)
    expect(screen.getByTitle('状态: idle')).toHaveClass('bg-gray-400')

    rerender(<AgentAvatar agent={agents[1]} />)
    expect(screen.getByTitle('状态: busy')).toHaveClass('bg-yellow-400')

    rerender(<AgentAvatar agent={agents[2]} />)
    expect(screen.getByTitle('状态: offline')).toHaveClass('bg-red-400')
  })
})

describe('RoleBadge', () => {
  it('应该渲染代码专家角色标签', () => {
    render(
      <RoleBadge role="code_expert" capabilities={['代码生成', '代码审查']} />
    )

    expect(screen.getByText('🔧')).toBeInTheDocument()
    expect(screen.getByText('代码专家')).toBeInTheDocument()
    expect(screen.getByText('(代码生成, 代码审查)')).toBeInTheDocument()
  })

  it('应该渲染架构师角色标签', () => {
    render(<RoleBadge role="architect" />)

    expect(screen.getByText('🏗️')).toBeInTheDocument()
    expect(screen.getByText('架构师')).toBeInTheDocument()
  })

  it('应该渲染文档专家角色标签', () => {
    render(<RoleBadge role="doc_expert" />)

    expect(screen.getByText('📝')).toBeInTheDocument()
    expect(screen.getByText('文档专家')).toBeInTheDocument()
  })

  it('应该渲染监督者角色标签', () => {
    render(<RoleBadge role="supervisor" />)

    expect(screen.getByText('👨‍💼')).toBeInTheDocument()
    expect(screen.getByText('任务调度器')).toBeInTheDocument()
  })

  it('应该限制显示的能力数量', () => {
    const capabilities = ['能力1', '能力2', '能力3', '能力4']
    render(<RoleBadge role="code_expert" capabilities={capabilities} />)

    // 应该只显示前两个能力
    expect(screen.getByText('(能力1, 能力2)')).toBeInTheDocument()
    expect(screen.queryByText('能力3')).not.toBeInTheDocument()
  })

  it('应该在没有能力时不显示能力列表', () => {
    render(<RoleBadge role="code_expert" capabilities={[]} />)

    expect(screen.queryByText(/\(/)).not.toBeInTheDocument()
  })
})

describe('StatusIndicator', () => {
  it('应该渲染活跃状态', () => {
    render(<StatusIndicator status="active" />)

    expect(screen.getByText('💬')).toBeInTheDocument()
    expect(screen.getByText('活跃')).toBeInTheDocument()

    const indicator = screen.getByText('活跃').parentElement
    expect(indicator).toHaveClass('bg-green-100', 'text-green-600')
  })

  it('应该渲染待机状态', () => {
    render(<StatusIndicator status="idle" />)

    expect(screen.getByText('💤')).toBeInTheDocument()
    expect(screen.getByText('待机')).toBeInTheDocument()

    const indicator = screen.getByText('待机').parentElement
    expect(indicator).toHaveClass('bg-gray-100', 'text-gray-600')
  })

  it('应该渲染忙碌状态', () => {
    render(<StatusIndicator status="busy" />)

    expect(screen.getByText('💭')).toBeInTheDocument()
    expect(screen.getByText('忙碌')).toBeInTheDocument()

    const indicator = screen.getByText('忙碌').parentElement
    expect(indicator).toHaveClass('bg-yellow-100', 'text-yellow-600')
  })

  it('应该渲染离线状态', () => {
    render(<StatusIndicator status="offline" />)

    expect(screen.getByText('🔴')).toBeInTheDocument()
    expect(screen.getByText('离线')).toBeInTheDocument()

    const indicator = screen.getByText('离线').parentElement
    expect(indicator).toHaveClass('bg-red-100', 'text-red-600')
  })

  it('应该支持不显示文字', () => {
    render(<StatusIndicator status="active" showText={false} />)

    expect(screen.getByText('💬')).toBeInTheDocument()
    expect(screen.queryByText('活跃')).not.toBeInTheDocument()
  })

  it('应该支持自定义样式类', () => {
    render(<StatusIndicator status="active" className="custom-class" />)

    const indicator = screen.getByText('活跃').parentElement
    expect(indicator).toHaveClass('custom-class')
  })
})
