import React, { useState, useRef, useEffect } from 'react'
import { Input, Button, message as antdMessage } from 'antd'
import type { TextAreaRef } from 'antd/es/input/TextArea'
import {
  SendOutlined,
  StopOutlined,
  WarningOutlined,
} from '@ant-design/icons'
import { MESSAGE_MAX_LENGTH, validateMessage } from '@/utils/validation'
import { useLocalDraft } from '@/hooks/useLocalDraft'

import { logger } from '../../utils/logger'
const { TextArea } = Input

interface MessageInputProps {
  onSendMessage: (message: string) => void
  onStop?: () => void
  loading: boolean
  placeholder?: string
  draftKey?: string
}

const DRAFT_STORAGE_PREFIX = 'ai-agent-chat-draft'
const QUICK_PROMPTS = [
  {
    label: '项目概览',
    value: '请简要概述这个系统的核心模块、数据流和主要能力。',
  },
  {
    label: '问题排查',
    value: '请根据错误日志定位可能原因，并给出优先级最高的修复步骤。',
  },
  {
    label: '测试用例',
    value: '请为一个API生成单元测试与集成测试的核心用例清单。',
  },
  {
    label: '性能优化',
    value: '请分析一个页面或接口的性能瓶颈，并给出可落地的优化方案。',
  },
  {
    label: '多智能体流程',
    value: '请设计一个多智能体协作流程来完成复杂任务，并说明分工。',
  },
  {
    label: '需求拆解',
    value: '请把一个需求拆分为可执行的迭代计划，并标注优先级。',
  },
]

const MessageInput: React.FC<MessageInputProps> = ({
  onSendMessage,
  onStop,
  loading,
  placeholder = '请输入你的问题...',
  draftKey,
}) => {
  const maxLength = MESSAGE_MAX_LENGTH
  const warnLength = maxLength - 200
  const draftStorageKey = `${DRAFT_STORAGE_PREFIX}:${draftKey || 'new'}`
  const [message, setMessage] = useLocalDraft(draftStorageKey, '')
  const [error, setError] = useState<string | null>(null)
  const textAreaRef = useRef<TextAreaRef>(null)
  const shortcutHint = onStop
    ? 'Enter发送 • Shift+Enter换行 • Esc停止生成'
    : 'Enter发送 • Shift+Enter换行'

  useEffect(() => {
    setError(null)
    requestAnimationFrame(() => {
      textAreaRef.current?.focus()
    })
  }, [draftStorageKey])

  const handleSend = () => {
    const trimmedMessage = message.trim()
    if (!trimmedMessage || loading) return

    // 验证消息
    const validation = validateMessage(trimmedMessage)
    if (!validation.isValid) {
      setError(validation.error || '消息验证失败')
      antdMessage.error(validation.error || '消息验证失败')
      return
    }

    // 清除错误状态
    setError(null)

    // 发送消息
    try {
      onSendMessage(trimmedMessage)
      setMessage('')
      textAreaRef.current?.focus()
    } catch (error) {
      logger.error('发送消息失败:', error)
      antdMessage.error('发送消息失败，请重试')
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value
    setMessage(value)

    if (!value.trim()) {
      if (error) setError(null)
      return
    }
    const validation = validateMessage(value.trim())
    setError(validation.isValid ? null : validation.error || '消息验证失败')
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Escape' && loading && onStop) {
      e.preventDefault()
      onStop()
      return
    }
    if (e.key !== 'Enter' || e.shiftKey || e.nativeEvent.isComposing) return
    e.preventDefault()
    handleSend()
  }

  const handlePromptClick = (value: string) => {
    if (loading) return
    const buildNextMessage = (prev: string) => {
      if (!prev.trim()) return value
      const separator = prev.endsWith('\n') ? '\n' : '\n\n'
      return `${prev}${separator}${value}`
    }
    setMessage(prev => {
      const next = buildNextMessage(prev)
      const validation = validateMessage(next)
      setError(validation.isValid ? null : validation.error || '消息验证失败')
      return next
    })
    requestAnimationFrame(() => {
      textAreaRef.current?.focus()
    })
  }

  return (
    <div className="border-t border-gray-100 p-4 bg-white">
      <div className="space-y-3">
        <div className="relative">
          <TextArea
            ref={textAreaRef}
            name="message"
            value={message}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            autoSize={{ minRows: 2, maxRows: 6 }}
            maxLength={maxLength}
            className={`resize-none rounded-xl border-gray-200 pr-14 ${error ? 'border-red-300' : 'focus:border-blue-400 focus:ring-2 focus:ring-blue-100'}`}
            status={error ? 'error' : undefined}
          />
          <Button
            type="primary"
            icon={loading ? <StopOutlined /> : <SendOutlined />}
            onClick={loading ? onStop : handleSend}
            disabled={loading ? !onStop : !message.trim() || !!error}
            className="absolute right-2 bottom-2 h-8 w-8 min-w-8 rounded-lg flex items-center justify-center p-0 bg-blue-500 hover:bg-blue-600 border-none shadow-sm"
            shape="circle"
            size="small"
          />
        </div>

        {!message.trim() && !loading && (
          <div className="flex flex-wrap gap-2">
            {QUICK_PROMPTS.map(prompt => (
              <Button
                key={prompt.label}
                size="small"
                type="default"
                onClick={() => handlePromptClick(prompt.value)}
                className="bg-gray-50"
              >
                {prompt.label}
              </Button>
            ))}
          </div>
        )}

        <div className="flex justify-between items-center text-xs text-gray-400">
          <span>{shortcutHint}</span>
          <span
            className={`${message.length > warnLength ? 'text-orange-500' : ''}`}
          >
            {message.length}/{maxLength}
          </span>
        </div>

        {error && (
          <div className="flex items-center space-x-2 text-red-500 text-sm bg-red-50 p-2 rounded-lg">
            <WarningOutlined />
            <span>{error}</span>
          </div>
        )}
      </div>
    </div>
  )
}

export default MessageInput
