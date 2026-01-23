import { render, screen } from '@testing-library/react'
import { ConfigProvider } from 'antd'
import MessageItem from '../../../src/components/conversation/MessageItem'
import { Message } from '../../../src/types'

// Mock MarkdownRenderer
vi.mock('../../../src/components/ui/MarkdownRenderer', () => ({
  default: ({ content }: { content: string }) => (
    <div data-testid="markdown">{content}</div>
  ),
}))

const renderWithProviders = (ui: React.ReactElement) => {
  return render(<ConfigProvider>{ui}</ConfigProvider>)
}

describe('MessageItem', () => {
  const mockUserMessage: Message = {
    id: '1',
    content: 'Hello, AI!',
    role: 'user',
    timestamp: '2023-12-01T10:00:00Z',
  }

  const mockAssistantMessage: Message = {
    id: '2',
    content: 'Hello! How can I help you?',
    role: 'assistant',
    timestamp: '2023-12-01T10:01:00Z',
  }

  it('renders user message correctly', () => {
    renderWithProviders(<MessageItem message={mockUserMessage} />)

    expect(screen.getByText('Hello, AI!')).toBeInTheDocument()
    expect(screen.getByText('用户')).toBeInTheDocument()
  })

  it('renders assistant message correctly', () => {
    renderWithProviders(<MessageItem message={mockAssistantMessage} />)

    expect(screen.getByTestId('markdown')).toHaveTextContent(
      'Hello! How can I help you?'
    )
    expect(screen.getByText('AI助手')).toBeInTheDocument()
  })

  it('displays timestamp when showTime is true', () => {
    renderWithProviders(
      <MessageItem message={mockUserMessage} showTime={true} />
    )

    // Should display time format (actual format is 18:00:00)
    expect(screen.getByText(/18:00/)).toBeInTheDocument()
  })

  it('applies correct styling for different roles', () => {
    const { rerender } = renderWithProviders(
      <MessageItem message={mockUserMessage} />
    )

    // User message should be right-aligned (check parent container)
    const userContainer = screen
      .getByText('Hello, AI!')
      .closest('.message-item')
    expect(userContainer).toHaveClass('justify-end')

    rerender(
      <ConfigProvider>
        <MessageItem message={mockAssistantMessage} />
      </ConfigProvider>
    )

    // Assistant message should be left-aligned
    const assistantContainer = screen
      .getByTestId('markdown')
      .closest('.message-item')
    expect(assistantContainer).toHaveClass('justify-start')
  })
})
