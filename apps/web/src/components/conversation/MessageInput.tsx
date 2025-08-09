import React, { useState, useRef } from 'react'
import { Input, Button, message as antdMessage } from 'antd'
import { SendOutlined, LoadingOutlined, WarningOutlined } from '@ant-design/icons'
import { validateMessage } from '@/utils/validation'

const { TextArea } = Input

interface MessageInputProps {
  onSendMessage: (message: string) => void
  loading: boolean
  placeholder?: string
}

const MessageInput: React.FC<MessageInputProps> = ({
  onSendMessage,
  loading,
  placeholder = '请输入你的问题...',
}) => {
  const [message, setMessage] = useState('')
  const [error, setError] = useState<string | null>(null)
  const textAreaRef = useRef<any>(null)

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
      console.error('发送消息失败:', error)
      antdMessage.error('发送消息失败，请重试')
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value
    setMessage(value)
    
    // 实时验证
    if (error && value.trim()) {
      const validation = validateMessage(value.trim())
      if (validation.isValid) {
        setError(null)
      }
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="border-t border-gray-100 p-4 bg-white">
      <div className="space-y-3">
        <div className="relative">
          <TextArea
            ref={textAreaRef}
            value={message}
            onChange={handleInputChange}
            onKeyPress={handleKeyPress}
            placeholder={placeholder}
            autoSize={{ minRows: 2, maxRows: 6 }}
            disabled={loading}
            className={`resize-none rounded-xl border-gray-200 pr-14 ${error ? 'border-red-300' : 'focus:border-blue-400 focus:ring-2 focus:ring-blue-100'}`}
            status={error ? 'error' : undefined}
          />
          <Button
            type="primary"
            icon={loading ? <LoadingOutlined /> : <SendOutlined />}
            onClick={handleSend}
            disabled={!message.trim() || loading || !!error}
            className="absolute right-2 bottom-2 h-8 w-8 min-w-8 rounded-lg flex items-center justify-center p-0 bg-blue-500 hover:bg-blue-600 border-none shadow-sm"
            shape="circle"
            size="small"
          />
        </div>
        
        <div className="flex justify-between items-center text-xs text-gray-400">
          <span>Enter发送 • Shift+Enter换行</span>
          <span className={`${message.length > 1800 ? 'text-orange-500' : ''}`}>
            {message.length}/2000
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