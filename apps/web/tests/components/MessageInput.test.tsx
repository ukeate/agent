import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import MessageInput from '../../src/components/conversation/MessageInput'

// Mock antd message
vi.mock('antd', async () => {
  const actual = await vi.importActual('antd')
  return {
    ...actual,
    message: {
      error: vi.fn(),
      success: vi.fn(),
      info: vi.fn(),
    }
  }
})

describe('MessageInput Component', () => {
  const mockOnSendMessage = vi.fn()
  
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders correctly', () => {
    render(
      <MessageInput 
        onSendMessage={mockOnSendMessage} 
        loading={false} 
      />
    )
    
    expect(screen.getByPlaceholderText('请输入你的问题...')).toBeInTheDocument()
    expect(screen.getByText('发送')).toBeInTheDocument()
  })

  it('calls onSendMessage when send button is clicked', async () => {
    render(
      <MessageInput 
        onSendMessage={mockOnSendMessage} 
        loading={false} 
      />
    )
    
    const textarea = screen.getByPlaceholderText('请输入你的问题...')
    const sendButton = screen.getByText('发送')
    
    fireEvent.change(textarea, { target: { value: 'Hello AI' } })
    fireEvent.click(sendButton)
    
    await waitFor(() => {
      expect(mockOnSendMessage).toHaveBeenCalledWith('Hello AI')
    })
  })

  it('prevents sending empty messages', () => {
    render(
      <MessageInput 
        onSendMessage={mockOnSendMessage} 
        loading={false} 
      />
    )
    
    const sendButton = screen.getByText('发送')
    
    fireEvent.click(sendButton)
    
    expect(mockOnSendMessage).not.toHaveBeenCalled()
  })

  it('disables input and button when loading', () => {
    render(
      <MessageInput 
        onSendMessage={mockOnSendMessage} 
        loading={true} 
      />
    )
    
    const textarea = screen.getByPlaceholderText('请输入你的问题...')
    const sendButton = screen.getByRole('button', { name: /发送中/i })
    
    expect(textarea).toBeDisabled()
    expect(sendButton).toBeDisabled()
  })

  it('shows character count', () => {
    render(
      <MessageInput 
        onSendMessage={mockOnSendMessage} 
        loading={false} 
      />
    )
    
    const textarea = screen.getByPlaceholderText('请输入你的问题...')
    
    fireEvent.change(textarea, { target: { value: 'Hello' } })
    
    expect(screen.getByText('5/2000')).toBeInTheDocument()
  })

  it('validates message length', async () => {
    render(
      <MessageInput 
        onSendMessage={mockOnSendMessage} 
        loading={false} 
      />
    )
    
    const textarea = screen.getByPlaceholderText('请输入你的问题...')
    const sendButton = screen.getByText('发送')
    
    // Create a message longer than 2000 characters
    const longMessage = 'a'.repeat(2001)
    fireEvent.change(textarea, { target: { value: longMessage } })
    fireEvent.click(sendButton)
    
    await waitFor(() => {
      expect(screen.getByText('消息长度不能超过2000个字符')).toBeInTheDocument()
    })
    
    expect(mockOnSendMessage).not.toHaveBeenCalled()
  })

  it('sends message on Enter key press', async () => {
    render(
      <MessageInput 
        onSendMessage={mockOnSendMessage} 
        loading={false} 
      />
    )
    
    const textarea = screen.getByPlaceholderText('请输入你的问题...')
    
    fireEvent.change(textarea, { target: { value: 'Hello AI' } })
    fireEvent.keyPress(textarea, { key: 'Enter', code: 'Enter', charCode: 13 })
    
    await waitFor(() => {
      expect(mockOnSendMessage).toHaveBeenCalledWith('Hello AI')
    })
  })

  it('does not send message on Shift+Enter', () => {
    render(
      <MessageInput 
        onSendMessage={mockOnSendMessage} 
        loading={false} 
      />
    )
    
    const textarea = screen.getByPlaceholderText('请输入你的问题...')
    
    fireEvent.change(textarea, { target: { value: 'Hello AI' } })
    fireEvent.keyPress(textarea, { 
      key: 'Enter', 
      code: 'Enter', 
      charCode: 13, 
      shiftKey: true 
    })
    
    expect(mockOnSendMessage).not.toHaveBeenCalled()
  })
})