import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { Message, Conversation } from '../types'

interface ConversationState {
  // 当前对话
  currentConversation: Conversation | null
  
  // 对话历史
  conversations: Conversation[]
  
  // 当前消息列表
  messages: Message[]
  
  // 加载状态
  loading: boolean
  
  // 错误状态
  error: string | null

  // Actions
  setCurrentConversation: (conversation: Conversation | null) => void
  addMessage: (message: Message) => void
  addMessages: (messages: Message[]) => void
  updateLastMessage: (content: string) => void
  clearMessages: () => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  
  // 对话管理
  createNewConversation: () => void
  saveConversation: () => void
  loadConversation: (conversationId: string) => void
  deleteConversation: (conversationId: string) => void
  
  // 消息管理
  deleteMessage: (messageId: string) => void
  updateMessage: (messageId: string, updates: Partial<Message>) => void
}

export const useConversationStore = create<ConversationState>()(
  persist(
    (set, get) => ({
      // 初始状态
      currentConversation: null,
      conversations: [],
      messages: [],
      loading: false,
      error: null,

      // 基础状态管理
      setCurrentConversation: (conversation) => {
        set({ 
          currentConversation: conversation,
          messages: conversation?.messages || [],
          error: null 
        })
      },

      addMessage: (message) => {
        set((state) => {
          const newMessages = [...state.messages, message]
          const updatedConversation = state.currentConversation ? {
            ...state.currentConversation,
            messages: newMessages,
            updatedAt: new Date().toISOString(),
          } : null
          
          // 同时更新conversations数组中的对话
          const updatedConversations = state.currentConversation 
            ? state.conversations.map(conv => 
                conv.id === state.currentConversation!.id 
                  ? updatedConversation!
                  : conv
              )
            : state.conversations
            
          return {
            messages: newMessages,
            currentConversation: updatedConversation,
            conversations: updatedConversations,
          }
        })
      },

      addMessages: (messages) => {
        set((state) => {
          const newMessages = [...state.messages, ...messages]
          const updatedConversation = state.currentConversation ? {
            ...state.currentConversation,
            messages: newMessages,
            updatedAt: new Date().toISOString(),
          } : null
          
          // 同时更新conversations数组中的对话
          const updatedConversations = state.currentConversation 
            ? state.conversations.map(conv => 
                conv.id === state.currentConversation!.id 
                  ? updatedConversation!
                  : conv
              )
            : state.conversations
            
          return {
            messages: newMessages,
            currentConversation: updatedConversation,
            conversations: updatedConversations,
          }
        })
      },

      updateLastMessage: (content) => {
        set((state) => {
          if (state.messages.length === 0) return state
          
          const newMessages = [...state.messages]
          const lastMessage = newMessages[newMessages.length - 1]
          newMessages[newMessages.length - 1] = {
            ...lastMessage,
            content: lastMessage.content + content,
          }
          
          const updatedConversation = state.currentConversation ? {
            ...state.currentConversation,
            messages: newMessages,
            updatedAt: new Date().toISOString(),
          } : null
          
          // 同时更新conversations数组中的对话
          const updatedConversations = state.currentConversation 
            ? state.conversations.map(conv => 
                conv.id === state.currentConversation!.id 
                  ? updatedConversation!
                  : conv
              )
            : state.conversations
          
          return {
            messages: newMessages,
            currentConversation: updatedConversation,
            conversations: updatedConversations,
          }
        })
      },

      clearMessages: () => {
        set({ 
          messages: [],
          currentConversation: null,
          error: null 
        })
      },

      setLoading: (loading) => set({ loading }),

      setError: (error) => set({ error }),

      // 对话管理
      createNewConversation: () => {
        const newConversation: Conversation = {
          id: `conv-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          title: '新对话',
          messages: [],
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        }
        
        set((state) => ({
          currentConversation: newConversation,
          conversations: [newConversation, ...state.conversations],
          messages: [],
          error: null,
        }))
      },

      saveConversation: () => {
        set((state) => {
          if (!state.currentConversation) return state
          
          const updatedConversation = {
            ...state.currentConversation,
            messages: state.messages,
            updatedAt: new Date().toISOString(),
            title: state.messages.length > 0 
              ? state.messages[0].content.slice(0, 30) + '...'
              : '新对话',
          }
          
          const conversationIndex = state.conversations.findIndex(
            c => c.id === updatedConversation.id
          )
          
          const newConversations = [...state.conversations]
          if (conversationIndex >= 0) {
            newConversations[conversationIndex] = updatedConversation
          } else {
            newConversations.unshift(updatedConversation)
          }
          
          return {
            currentConversation: updatedConversation,
            conversations: newConversations,
          }
        })
      },

      loadConversation: (conversationId) => {
        const state = get()
        const conversation = state.conversations.find(c => c.id === conversationId)
        if (conversation) {
          set({
            currentConversation: conversation,
            messages: conversation.messages,
            error: null,
          })
        }
      },

      deleteConversation: (conversationId) => {
        set((state) => {
          const newConversations = state.conversations.filter(
            c => c.id !== conversationId
          )
          
          return {
            conversations: newConversations,
            currentConversation: state.currentConversation?.id === conversationId 
              ? null 
              : state.currentConversation,
            messages: state.currentConversation?.id === conversationId 
              ? [] 
              : state.messages,
          }
        })
      },

      // 消息管理
      deleteMessage: (messageId) => {
        set((state) => {
          const newMessages = state.messages.filter(m => m.id !== messageId)
          return {
            messages: newMessages,
            currentConversation: state.currentConversation ? {
              ...state.currentConversation,
              messages: newMessages,
              updatedAt: new Date().toISOString(),
            } : null,
          }
        })
      },

      updateMessage: (messageId, updates) => {
        set((state) => {
          const newMessages = state.messages.map(m => 
            m.id === messageId ? { ...m, ...updates } : m
          )
          
          return {
            messages: newMessages,
            currentConversation: state.currentConversation ? {
              ...state.currentConversation,
              messages: newMessages,
              updatedAt: new Date().toISOString(),
            } : null,
          }
        })
      },
    }),
    {
      name: 'conversation-store',
      // 持久化对话历史和当前对话ID
      partialize: (state) => ({
        conversations: state.conversations,
        currentConversationId: state.currentConversation?.id || null,
      }),
      // 恢复状态时重新设置当前对话
      onRehydrateStorage: () => (state) => {
        if (state) {
          const currentConversationId = (state as any).currentConversationId
          if (currentConversationId) {
            const conversation = state.conversations.find(c => c.id === currentConversationId)
            if (conversation) {
              state.currentConversation = conversation
              state.messages = conversation.messages
            }
          }
        }
      },
    }
  )
)