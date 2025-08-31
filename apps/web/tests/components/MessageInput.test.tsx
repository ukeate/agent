import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import MessageInput from '../../src/components/conversation/MessageInput'

// Mock antd message
mockFn()mock('antd', async () => {
  const actual = await mockFn()importActual('antd')
  return {
    ...actual,
    message: {
      error: mockFn()fn(),
      success: mockFn()fn(),
      info: mockFn()fn(),
    }
  }
})

describe('MessageInput Component', () => {
  const mockOnSendMessage = mockFn()fn()
  
  beforeEach(() => {
    mockFn()clearAllMocks()
  })

  it('renders correctly', () => {
    render(
      <MessageInput 
        onSendMessage={mockOnSendMessage} 
        loading={false} 
      />
    )
    
    expect(screen.getByPlaceholderText('请输入你的问题...')).toBeInTheDocument()
    expect(screen.getByRole('button')).toBeInTheDocument()
  })

  it('calls onSendMessage when send button is clicked', async () => {
    render(
      <MessageInput 
        onSendMessage={mockOnSendMessage} 
        loading={false} 
      />
    )
    
    const textarea = screen.getByPlaceholderText('请输入你的问题...')
    const sendButton = screen.getByRole('button')
    
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
    
    const sendButton = screen.getByRole('button')
    
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
    const sendButton = screen.getByRole('button')
    
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
    const sendButton = screen.getByRole('button')
    
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