import { render, screen, fireEvent } from '@testing-library/react'
import { ConfigProvider } from 'antd'
import { BrowserRouter } from 'react-router-dom'
import ConversationHistory from '../../../src/components/conversation/ConversationHistory'
import { Conversation } from '../../../src/types'
import { useConversationStore } from '../../../src/stores/conversationStore'

// Mock zustand store
vi.mock('../../../src/stores/conversationStore', () => ({
  useConversationStore: vi.fn(),
}))

const mockUseConversationStore = vi.mocked(useConversationStore)

const renderWithProviders = (ui: React.ReactElement) => {
  return render(
    <BrowserRouter>
      <ConfigProvider>{ui}</ConfigProvider>
    </BrowserRouter>
  )
}

describe('ConversationHistory', () => {
  const mockConversations: Conversation[] = [
    {
      id: '1',
      title: 'Test Conversation 1',
      messages: [
        {
          id: 'msg1',
          content: 'Hello',
          role: 'user',
          timestamp: '2023-12-01T10:00:00Z',
        },
        {
          id: 'msg2',
          content: 'Hi there!',
          role: 'agent',
          timestamp: '2023-12-01T10:01:00Z',
        },
      ],
      createdAt: '2023-12-01T10:00:00Z',
      updatedAt: '2023-12-01T10:30:00Z',
    },
    {
      id: '2',
      title: 'Test Conversation 2',
      messages: [
        {
          id: 'msg3',
          content: '多智能体测试',
          role: 'user',
          timestamp: '2023-12-01T09:00:00Z',
        },
        {
          id: 'msg4',
          content: '这是多智能体响应',
          role: 'agent',
          timestamp: '2023-12-01T09:01:00Z',
        },
        {
          id: 'msg5',
          content: '继续对话',
          role: 'user',
          timestamp: '2023-12-01T09:02:00Z',
        },
      ],
      createdAt: '2023-12-01T09:00:00Z',
      updatedAt: '2023-12-01T09:30:00Z',
    },
  ]

  const mockOnSelectConversation = vi.fn()
  const mockDeleteConversation = vi.fn()
  const mockRefreshConversations = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    mockRefreshConversations.mockResolvedValue({
      items: mockConversations,
      hasMore: false,
    })

    // Default mock store state
    mockUseConversationStore.mockReturnValue({
      conversations: mockConversations,
      deleteConversation: mockDeleteConversation,
      refreshConversations: mockRefreshConversations,
      currentConversation: null,
      messages: [],
      loading: false,
      error: null,
      historyLoading: false,
      historyError: null,
      setHistoryError: vi.fn(),
      setCurrentConversation: vi.fn(),
      addMessage: vi.fn(),
      addMessages: vi.fn(),
      updateLastMessage: vi.fn(),
      clearMessages: vi.fn(),
      setLoading: vi.fn(),
      setError: vi.fn(),
      createNewConversation: vi.fn(),
      saveConversation: vi.fn(),
      loadConversation: vi.fn(),
      deleteMessage: vi.fn(),
      updateMessage: vi.fn(),
    })
  })

  it('renders empty state when no conversations', () => {
    // Mock store with empty conversations
    mockRefreshConversations.mockResolvedValue({ items: [], hasMore: false })
    mockUseConversationStore.mockReturnValue({
      conversations: [],
      deleteConversation: mockDeleteConversation,
      refreshConversations: mockRefreshConversations,
      currentConversation: null,
      messages: [],
      loading: false,
      error: null,
      historyLoading: false,
      historyError: null,
      setHistoryError: vi.fn(),
      setCurrentConversation: vi.fn(),
      addMessage: vi.fn(),
      addMessages: vi.fn(),
      updateLastMessage: vi.fn(),
      clearMessages: vi.fn(),
      setLoading: vi.fn(),
      setError: vi.fn(),
      createNewConversation: vi.fn(),
      saveConversation: vi.fn(),
      loadConversation: vi.fn(),
      deleteMessage: vi.fn(),
      updateMessage: vi.fn(),
    })

    renderWithProviders(
      <ConversationHistory
        visible={true}
        onSelectConversation={mockOnSelectConversation}
      />
    )

    expect(screen.getByText(/暂无对话历史/)).toBeInTheDocument()
  })

  it('renders conversation list', () => {
    renderWithProviders(
      <ConversationHistory
        visible={true}
        onSelectConversation={mockOnSelectConversation}
      />
    )

    expect(screen.getByText('Test Conversation 1')).toBeInTheDocument()
    expect(screen.getByText('Test Conversation 2')).toBeInTheDocument()
    expect(screen.getByText(/共 2 个对话/)).toBeInTheDocument()
  })

  it('shows message counts correctly', () => {
    renderWithProviders(
      <ConversationHistory
        visible={true}
        onSelectConversation={mockOnSelectConversation}
      />
    )

    // Should show message counts
    expect(screen.getAllByText('2 条消息')).toHaveLength(1)
    expect(screen.getByText('3 条消息')).toBeInTheDocument()
    expect(screen.getByText('1 次提问')).toBeInTheDocument()
    expect(screen.getByText('2 次提问')).toBeInTheDocument()
  })

  it('handles conversation selection', () => {
    renderWithProviders(
      <ConversationHistory
        visible={true}
        onSelectConversation={mockOnSelectConversation}
      />
    )

    fireEvent.click(screen.getByText('Test Conversation 1'))

    expect(mockOnSelectConversation).toHaveBeenCalledWith(mockConversations[0])
  })

  it('does not render when not visible', () => {
    const { container } = renderWithProviders(
      <ConversationHistory
        visible={false}
        onSelectConversation={mockOnSelectConversation}
      />
    )

    expect(container.firstChild).toBeNull()
  })
})
