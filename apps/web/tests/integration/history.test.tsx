import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useConversationStore } from '../../src/stores/conversationStore'
import { Message } from '../../src/types'

// 模拟localStorage
const localStorageMock = {
  getItem: mockFn()fn(),
  setItem: mockFn()fn(),
  removeItem: mockFn()fn(),
  clear: mockFn()fn(),
}
Object.defineProperty(window, 'localStorage', {
  value: localStorageMock
})

describe('聊天历史记录集成测试', () => {
  beforeEach(() => {
    // 清理store状态
    useConversationStore.getState().clearMessages()
    useConversationStore.setState({
      currentConversation: null,
      conversations: [],
      messages: [],
    })
    mockFn()clearAllMocks()
  })

  it('应该能够创建新对话并保存消息', () => {
    const store = useConversationStore.getState()
    
    // 创建新对话
    store.createNewConversation()
    
    // 验证对话已创建
    const state = useConversationStore.getState()
    expect(state.currentConversation).toBeTruthy()
    expect(state.currentConversation?.title).toBe('新对话')
    expect(state.conversations).toHaveLength(1)
  })

  it('应该能够添加消息到当前对话', () => {
    const store = useConversationStore.getState()
    
    // 创建新对话
    store.createNewConversation()
    
    // 添加用户消息
    const userMessage: Message = {
      id: 'test-user-1',
      content: '这是一条测试消息',
      role: 'user',
      timestamp: new Date().toISOString(),
    }
    
    store.addMessage(userMessage)
    
    // 添加AI响应
    const agentMessage: Message = {
      id: 'test-agent-1',
      content: '这是AI的回复',
      role: 'agent',
      timestamp: new Date().toISOString(),
    }
    
    store.addMessage(agentMessage)
    
    // 验证消息已添加
    const state = useConversationStore.getState()
    expect(state.messages).toHaveLength(2)
    expect(state.messages[0]).toEqual(userMessage)
    expect(state.messages[1]).toEqual(agentMessage)
    
    // 保存对话以更新标题
    store.saveConversation()
    
    // 验证对话标题已更新
    const updatedState = useConversationStore.getState()
    expect(updatedState.currentConversation?.title).toBe('这是一条测试消息...')
  })

  it('应该能够保存对话到历史记录', () => {
    const store = useConversationStore.getState()
    
    // 创建新对话并添加消息
    store.createNewConversation()
    
    const userMessage: Message = {
      id: 'test-user-1',
      content: '测试历史记录保存',
      role: 'user',
      timestamp: new Date().toISOString(),
    }
    
    store.addMessage(userMessage)
    
    // 保存对话
    store.saveConversation()
    
    // 验证对话已保存到历史记录
    const state = useConversationStore.getState()
    expect(state.conversations).toHaveLength(1)
    expect(state.conversations[0].messages).toHaveLength(1)
    expect(state.conversations[0].title).toBe('测试历史记录保存...')
  })

  it('应该能够加载历史对话', () => {
    const store = useConversationStore.getState()
    
    // 创建并保存一个对话
    store.createNewConversation()
    const conversationId = useConversationStore.getState().currentConversation!.id
    
    const testMessage: Message = {
      id: 'test-1',
      content: '历史对话消息',
      role: 'user',
      timestamp: new Date().toISOString(),
    }
    
    store.addMessage(testMessage)
    store.saveConversation()
    
    // 清空当前状态
    store.clearMessages()
    expect(useConversationStore.getState().messages).toHaveLength(0)
    
    // 加载历史对话
    store.loadConversation(conversationId)
    
    // 验证对话已加载
    const state = useConversationStore.getState()
    expect(state.currentConversation?.id).toBe(conversationId)
    expect(state.messages).toHaveLength(1)
    expect(state.messages[0].content).toBe('历史对话消息')
  })

  it('应该能够删除历史对话', () => {
    const store = useConversationStore.getState()
    
    // 创建并保存第一个对话
    store.createNewConversation()
    const firstConvId = useConversationStore.getState().currentConversation!.id
    store.addMessage({
      id: 'msg-1',
      content: '第一个对话',
      role: 'user',
      timestamp: new Date().toISOString(),
    })
    store.saveConversation()
    
    // 创建并保存第二个对话
    store.createNewConversation()
    const secondConvId = useConversationStore.getState().currentConversation!.id
    store.addMessage({
      id: 'msg-2',
      content: '第二个对话',
      role: 'user',
      timestamp: new Date().toISOString(),
    })
    store.saveConversation()
    
    // 验证有两个对话
    let state = useConversationStore.getState()
    expect(state.conversations).toHaveLength(2)
    
    // 删除第一个对话
    store.deleteConversation(firstConvId)
    
    // 验证对话已删除
    state = useConversationStore.getState()
    expect(state.conversations).toHaveLength(1)
    expect(state.conversations[0].id).toBe(secondConvId)
  })

  it('应该能够更新流式消息内容', () => {
    const store = useConversationStore.getState()
    
    // 创建新对话
    store.createNewConversation()
    
    // 添加一个空的AI消息（模拟流式响应开始）
    const agentMessage: Message = {
      id: 'agent-stream',
      content: '',
      role: 'agent',
      timestamp: new Date().toISOString(),
    }
    
    store.addMessage(agentMessage)
    
    // 模拟流式更新
    store.updateLastMessage('Hello')
    store.updateLastMessage(' world')
    store.updateLastMessage('!')
    
    // 验证消息内容已累积更新
    const state = useConversationStore.getState()
    expect(state.messages).toHaveLength(1)
    expect(state.messages[0].content).toBe('Hello world!')
  })
})