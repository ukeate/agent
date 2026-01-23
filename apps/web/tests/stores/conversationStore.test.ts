import { describe, it, expect, beforeEach } from 'vitest'
import { useConversationStore } from '../../src/stores/conversationStore'
import { Message } from '../../src/types'

// Mock localStorage for testing
const localStorageMock = {
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {},
  clear: () => {},
}
Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
})

describe('ConversationStore', () => {
  beforeEach(() => {
    // Reset store state before each test
    useConversationStore.setState({
      currentConversation: null,
      conversations: [],
      messages: [],
      loading: false,
      error: null,
      historyLoading: false,
      historyError: null,
    })
  })

  it('initializes with empty state', () => {
    const state = useConversationStore.getState()

    expect(state.currentConversation).toBeNull()
    expect(state.conversations).toEqual([])
    expect(state.messages).toEqual([])
    expect(state.loading).toBe(false)
    expect(state.error).toBeNull()
    expect(state.historyLoading).toBe(false)
    expect(state.historyError).toBeNull()
  })

  it('adds messages correctly', () => {
    const { addMessage } = useConversationStore.getState()

    const message: Message = {
      id: 'msg-1',
      content: 'Hello',
      role: 'user',
      timestamp: '2023-01-01T00:00:00Z',
    }

    addMessage(message)

    const state = useConversationStore.getState()
    expect(state.messages).toHaveLength(1)
    expect(state.messages[0]).toEqual(message)
  })

  it('clears messages correctly', () => {
    const { addMessage, clearMessages } = useConversationStore.getState()

    const message: Message = {
      id: 'msg-1',
      content: 'Hello',
      role: 'user',
      timestamp: '2023-01-01T00:00:00Z',
    }

    addMessage(message)
    expect(useConversationStore.getState().messages).toHaveLength(1)

    clearMessages()
    const state = useConversationStore.getState()
    expect(state.messages).toEqual([])
    expect(state.currentConversation).toBeNull()
    expect(state.error).toBeNull()
  })

  it('updates loading state', () => {
    const { setLoading } = useConversationStore.getState()

    setLoading(true)
    expect(useConversationStore.getState().loading).toBe(true)

    setLoading(false)
    expect(useConversationStore.getState().loading).toBe(false)
  })

  it('sets error state', () => {
    const { setError } = useConversationStore.getState()

    setError('Test error')
    expect(useConversationStore.getState().error).toBe('Test error')

    setError(null)
    expect(useConversationStore.getState().error).toBeNull()
  })

  it('creates new conversation', () => {
    const { createNewConversation } = useConversationStore.getState()

    createNewConversation()

    const state = useConversationStore.getState()
    expect(state.currentConversation).not.toBeNull()
    expect(state.currentConversation?.title).toBe('新对话')
    expect(state.conversations).toHaveLength(1)
    expect(state.messages).toEqual([])
  })

  it('loads conversation correctly', () => {
    const { createNewConversation, loadConversation, addMessage } =
      useConversationStore.getState()

    // Create a conversation with messages
    createNewConversation()
    const conversationId =
      useConversationStore.getState().currentConversation!.id

    const message: Message = {
      id: 'msg-1',
      content: 'Hello',
      role: 'user',
      timestamp: '2023-01-01T00:00:00Z',
    }

    addMessage(message)

    // Verify the conversation in the conversations list was updated
    const stateBefore = useConversationStore.getState()
    const conversationBefore = stateBefore.conversations.find(
      c => c.id === conversationId
    )
    expect(conversationBefore?.messages).toHaveLength(1)
    expect(stateBefore.messages).toHaveLength(1)

    // Clear current state to simulate switching conversations
    useConversationStore.setState({
      currentConversation: null,
      messages: [],
    })

    // Load the first conversation
    loadConversation(conversationId)

    const stateAfterLoad = useConversationStore.getState()
    expect(stateAfterLoad.currentConversation?.id).toBe(conversationId)
    expect(stateAfterLoad.messages).toHaveLength(1)
    expect(stateAfterLoad.messages[0]).toEqual(message)
  })

  it('deletes conversation correctly', () => {
    const { createNewConversation, deleteConversation } =
      useConversationStore.getState()

    createNewConversation()
    const conversationId =
      useConversationStore.getState().currentConversation!.id

    expect(useConversationStore.getState().conversations).toHaveLength(1)

    deleteConversation(conversationId)

    const state = useConversationStore.getState()
    expect(state.conversations).toEqual([])
    expect(state.currentConversation).toBeNull()
    expect(state.messages).toEqual([])
  })

  it('updates last message content', () => {
    const { addMessage, updateLastMessage } = useConversationStore.getState()

    const message: Message = {
      id: 'msg-1',
      content: 'Hello',
      role: 'agent',
      timestamp: '2023-01-01T00:00:00Z',
    }

    addMessage(message)
    updateLastMessage(' World')

    const state = useConversationStore.getState()
    expect(state.messages[0].content).toBe('Hello World')
  })

  it('adds multiple messages correctly', () => {
    const { addMessages } = useConversationStore.getState()

    const messages: Message[] = [
      {
        id: 'msg-1',
        content: 'Hello',
        role: 'user',
        timestamp: '2023-01-01T00:00:00Z',
      },
      {
        id: 'msg-2',
        content: 'Hi there',
        role: 'agent',
        timestamp: '2023-01-01T00:00:01Z',
      },
    ]

    addMessages(messages)

    const state = useConversationStore.getState()
    expect(state.messages).toHaveLength(2)
    expect(state.messages).toEqual(messages)
  })
})
