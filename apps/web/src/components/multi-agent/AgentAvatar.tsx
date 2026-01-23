import React from 'react'
import { Agent } from '../../stores/multiAgentStore'

interface AgentAvatarProps {
  agent: Agent
  size?: 'sm' | 'md' | 'lg'
  showStatus?: boolean
  className?: string
}

// 角色颜色映射
const ROLE_COLORS = {
  code_expert: 'bg-blue-500',
  architect: 'bg-green-500',
  doc_expert: 'bg-orange-500',
  supervisor: 'bg-purple-500',
} as const

// 角色图标映射
const ROLE_ICONS = {
  code_expert: '🔧',
  architect: '🏗️',
  doc_expert: '📝',
  supervisor: '👨‍💼',
} as const

// 状态颜色映射
const STATUS_COLORS = {
  active: 'bg-green-400',
  idle: 'bg-gray-400',
  busy: 'bg-yellow-400',
  offline: 'bg-red-400',
} as const

// 尺寸映射
const SIZE_CLASSES = {
  sm: 'w-8 h-8 text-sm',
  md: 'w-12 h-12 text-lg',
  lg: 'w-16 h-16 text-xl',
} as const

const STATUS_SIZE_CLASSES = {
  sm: 'w-2 h-2',
  md: 'w-3 h-3',
  lg: 'w-4 h-4',
} as const

export const AgentAvatar: React.FC<AgentAvatarProps> = ({
  agent,
  size = 'md',
  showStatus = true,
  className = '',
}) => {
  const baseColor = ROLE_COLORS[agent.role]
  const icon = ROLE_ICONS[agent.role]
  const statusColor = STATUS_COLORS[agent.status]
  const sizeClass = SIZE_CLASSES[size]
  const statusSizeClass = STATUS_SIZE_CLASSES[size]

  return (
    <div className={`relative inline-block ${className}`}>
      {/* 主头像 */}
      <div
        className={`
          ${sizeClass} ${baseColor}
          rounded-full flex items-center justify-center
          text-white font-medium shadow-lg
          transition-transform hover:scale-105
        `}
        title={`${agent.name} (${agent.role})`}
      >
        <span className="select-none">{icon}</span>
      </div>

      {/* 状态指示器 */}
      {showStatus && (
        <div
          className={`
            absolute -bottom-1 -right-1
            ${statusSizeClass} ${statusColor}
            rounded-full border-2 border-white
          `}
          title={`状态: ${agent.status}`}
        />
      )}
    </div>
  )
}

// 角色标签组件
interface RoleBadgeProps {
  role: Agent['role']
  capabilities?: string[]
  className?: string
}

export const RoleBadge: React.FC<RoleBadgeProps> = ({
  role,
  capabilities = [],
  className = '',
}) => {
  const color = ROLE_COLORS[role]
  const icon = ROLE_ICONS[role]

  const roleNames = {
    code_expert: '代码专家',
    architect: '架构师',
    doc_expert: '文档专家',
    supervisor: '任务调度器',
  }

  return (
    <div
      className={`
        inline-flex items-center gap-2 px-3 py-1
        ${color} text-white text-sm font-medium
        rounded-full shadow-sm
        ${className}
      `}
    >
      <span>{icon}</span>
      <span>{roleNames[role]}</span>
      {capabilities.length > 0 && (
        <div className="text-xs opacity-80">
          ({capabilities.slice(0, 2).join(', ')})
        </div>
      )}
    </div>
  )
}

// 状态指示器组件
interface StatusIndicatorProps {
  status: Agent['status']
  showText?: boolean
  className?: string
}

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  showText = true,
  className = '',
}) => {
  const statusInfo = {
    active: {
      color: 'text-green-600',
      bg: 'bg-green-100',
      icon: '💬',
      text: '活跃',
    },
    idle: {
      color: 'text-gray-600',
      bg: 'bg-gray-100',
      icon: '💤',
      text: '待机',
    },
    busy: {
      color: 'text-yellow-600',
      bg: 'bg-yellow-100',
      icon: '💭',
      text: '忙碌',
    },
    offline: {
      color: 'text-red-600',
      bg: 'bg-red-100',
      icon: '🔴',
      text: '离线',
    },
  }

  const info = statusInfo[status]

  return (
    <div
      className={`
        inline-flex items-center gap-1 px-2 py-1
        ${info.bg} ${info.color}
        rounded-md text-xs font-medium
        ${className}
      `}
    >
      <span>{info.icon}</span>
      {showText && <span>{info.text}</span>}
    </div>
  )
}
