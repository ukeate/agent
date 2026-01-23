/**
 * 情感智能WebSocket Hook
 * 提供WebSocket连接管理、状态管理和上下文记忆功能
 */

import { buildWsUrl } from '../utils/apiBase'
import { useState, useEffect, useRef, useCallback } from 'react';
import { toast } from 'sonner';
import { logger } from '../utils/logger'
import {
  EmotionWebSocketService,
  ConnectionState,
  UnifiedEmotionalData,
  EmotionInputMessage,
  ModalityType,
  createEmotionWebSocketService,
  WebSocketConfig
} from '@/services/emotionWebSocketService';

// 会话状态接口
export interface SessionState {
  sessionId: string;
  userId: string;
  startTime: Date;
  lastActivity: Date;
  messageCount: number;
  emotionHistory: UnifiedEmotionalData[];
  contextMemory: ContextMemory[];
  isActive: boolean;
}

// 上下文记忆接口
export interface ContextMemory {
  id: string;
  type: 'emotion' | 'interaction' | 'preference' | 'event';
  content: any;
  importance: number; // 0-1的重要性评分
  timestamp: Date;
  associatedEmotions?: string[];
  tags?: string[];
  decay?: number; // 记忆衰减值
}

// Hook配置接口
export interface UseEmotionWebSocketConfig extends Omit<WebSocketConfig, 'userId'> {
  userId?: string;
  maxHistoryLength?: number;
  maxMemoryLength?: number;
  memoryDecayRate?: number;
  autoSaveSession?: boolean;
  storagePrefix?: string;
}

// Hook返回值接口
export interface UseEmotionWebSocketResult {
  // 连接状态
  connectionState: ConnectionState;
  isConnected: boolean;
  stats: any;
  
  // 会话管理
  session: SessionState | null;
  sessionId: string | null;
  
  // 情感数据
  currentEmotion: UnifiedEmotionalData | null;
  emotionHistory: UnifiedEmotionalData[];
  contextMemory: ContextMemory[];
  
  // 操作方法
  connect: () => Promise<void>;
  disconnect: () => void;
  sendEmotionInput: (input: EmotionInputMessage) => Promise<void>;
  sendTextEmotion: (text: string) => Promise<void>;
  sendAudioEmotion: (audioBlob: Blob) => Promise<void>;
  sendVideoEmotion: (videoBlob: Blob) => Promise<void>;
  sendImageEmotion: (imageFile: File) => Promise<void>;
  sendPhysiologicalData: (data: Record<string, any>) => Promise<void>;
  
  // 会话和记忆管理
  saveSession: () => void;
  loadSession: (sessionId: string) => Promise<boolean>;
  clearSession: () => void;
  addContextMemory: (memory: Omit<ContextMemory, 'id' | 'timestamp'>) => void;
  getRelevantMemories: (query: string, limit?: number) => ContextMemory[];
  updateMemoryImportance: (memoryId: string, importance: number) => void;
  
  // 错误和状态
  error: string | null;
  clearError: () => void;
}

/**
 * 情感WebSocket Hook
 */
export function useEmotionWebSocket(config: UseEmotionWebSocketConfig): UseEmotionWebSocketResult {
  // 配置默认值
  const finalConfig = {
    url: buildWsUrl('/ws'),
    userId: 'anonymous',
    maxHistoryLength: 100,
    maxMemoryLength: 1000,
    memoryDecayRate: 0.01,
    autoSaveSession: true,
    storagePrefix: 'emotion_session',
    autoReconnect: true,
    reconnectInterval: 5000,
    maxReconnectAttempts: 10,
    heartbeatInterval: 30000,
    ...config
  };

  // WebSocket服务引用
  const wsServiceRef = useRef<EmotionWebSocketService | null>(null);
  
  // 状态管理
  const [connectionState, setConnectionState] = useState<ConnectionState>(ConnectionState.DISCONNECTED);
  const [session, setSession] = useState<SessionState | null>(null);
  const [currentEmotion, setCurrentEmotion] = useState<UnifiedEmotionalData | null>(null);
  const [emotionHistory, setEmotionHistory] = useState<UnifiedEmotionalData[]>([]);
  const [contextMemory, setContextMemory] = useState<ContextMemory[]>([]);
  const [stats, setStats] = useState<any>({});
  const [error, setError] = useState<string | null>(null);

  // 内部状态
  const sessionIdRef = useRef<string | null>(null);
  const memoryDecayTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // 生成会话ID
  const generateSessionId = useCallback(() => {
    return `session_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }, []);

  // 生成记忆ID
  const generateMemoryId = useCallback(() => {
    return `memory_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }, []);

  // 初始化会话
  const initializeSession = useCallback(() => {
    const sessionId = generateSessionId();
    sessionIdRef.current = sessionId;

    const newSession: SessionState = {
      sessionId,
      userId: finalConfig.userId,
      startTime: new Date(),
      lastActivity: new Date(),
      messageCount: 0,
      emotionHistory: [],
      contextMemory: [],
      isActive: true
    };

    setSession(newSession);
    
    if (finalConfig.autoSaveSession) {
      localStorage.setItem(
        `${finalConfig.storagePrefix}_current`,
        JSON.stringify({
          sessionId,
          timestamp: new Date().toISOString()
        })
      );
    }

    return newSession;
  }, [finalConfig.userId, finalConfig.storagePrefix, finalConfig.autoSaveSession, generateSessionId]);

  // 更新会话活动
  const updateSessionActivity = useCallback(() => {
    if (session) {
      const updatedSession = {
        ...session,
        lastActivity: new Date(),
        messageCount: session.messageCount + 1
      };
      setSession(updatedSession);
    }
  }, [session]);

  // 添加情感数据到历史
  const addEmotionToHistory = useCallback((emotion: UnifiedEmotionalData) => {
    setEmotionHistory(prev => {
      const newHistory = [...prev, emotion];
      // 限制历史长度
      if (newHistory.length > finalConfig.maxHistoryLength) {
        newHistory.splice(0, newHistory.length - finalConfig.maxHistoryLength);
      }
      return newHistory;
    });

    // 自动创建情感记忆
    if (emotion.emotional_state || emotion.recognition_result) {
      const emotionMemory: Omit<ContextMemory, 'id' | 'timestamp'> = {
        type: 'emotion',
        content: {
          emotion_state: emotion.emotional_state,
          recognition_result: emotion.recognition_result,
          confidence: emotion.confidence
        },
        importance: emotion.confidence * 0.8, // 基于置信度计算重要性
        associatedEmotions: emotion.emotional_state ? [emotion.emotional_state.emotion] : [],
        tags: ['auto_generated', 'emotion_analysis']
      };

      addContextMemory(emotionMemory);
    }
  }, [finalConfig.maxHistoryLength]);

  // 添加上下文记忆
  const addContextMemory = useCallback((memory: Omit<ContextMemory, 'id' | 'timestamp'>) => {
    const newMemory: ContextMemory = {
      ...memory,
      id: generateMemoryId(),
      timestamp: new Date()
    };

    setContextMemory(prev => {
      const newMemories = [...prev, newMemory];
      
      // 按重要性排序并限制数量
      newMemories.sort((a, b) => b.importance - a.importance);
      if (newMemories.length > finalConfig.maxMemoryLength) {
        newMemories.splice(finalConfig.maxMemoryLength);
      }
      
      return newMemories;
    });
  }, [finalConfig.maxMemoryLength, generateMemoryId]);

  // 获取相关记忆
  const getRelevantMemories = useCallback((query: string, limit = 10): ContextMemory[] => {
    const queryLower = query.toLowerCase();
    
    return contextMemory
      .filter(memory => {
        // 简单的相关性匹配
        const contentStr = JSON.stringify(memory.content).toLowerCase();
        const tagsStr = memory.tags?.join(' ').toLowerCase() || '';
        
        return contentStr.includes(queryLower) || 
               tagsStr.includes(queryLower) ||
               memory.associatedEmotions?.some(emotion => 
                 emotion.toLowerCase().includes(queryLower)
               );
      })
      .sort((a, b) => {
        // 按重要性和时间排序
        const importanceDiff = b.importance - a.importance;
        if (Math.abs(importanceDiff) > 0.1) {
          return importanceDiff;
        }
        return b.timestamp.getTime() - a.timestamp.getTime();
      })
      .slice(0, limit);
  }, [contextMemory]);

  // 更新记忆重要性
  const updateMemoryImportance = useCallback((memoryId: string, importance: number) => {
    setContextMemory(prev => 
      prev.map(memory => 
        memory.id === memoryId 
          ? { ...memory, importance: Math.max(0, Math.min(1, importance)) }
          : memory
      )
    );
  }, []);

  // 记忆衰减处理
  const processMemoryDecay = useCallback(() => {
    const now = new Date();
    
    setContextMemory(prev => 
      prev.map(memory => {
        const ageInHours = (now.getTime() - memory.timestamp.getTime()) / (1000 * 60 * 60);
        const decayFactor = Math.exp(-finalConfig.memoryDecayRate * ageInHours);
        const newImportance = memory.importance * decayFactor;
        
        return {
          ...memory,
          importance: newImportance,
          decay: 1 - decayFactor
        };
      })
      .filter(memory => memory.importance > 0.01) // 移除重要性过低的记忆
      .sort((a, b) => b.importance - a.importance)
    );
  }, [finalConfig.memoryDecayRate]);

  // 保存会话
  const saveSession = useCallback(() => {
    if (!session || !finalConfig.autoSaveSession) return;

    const sessionData = {
      session,
      emotionHistory,
      contextMemory,
      timestamp: new Date().toISOString()
    };

    localStorage.setItem(
      `${finalConfig.storagePrefix}_${session.sessionId}`,
      JSON.stringify(sessionData)
    );

    toast.success('会话已保存');
  }, [session, emotionHistory, contextMemory, finalConfig.autoSaveSession, finalConfig.storagePrefix]);

  // 加载会话
  const loadSession = useCallback(async (sessionId: string): Promise<boolean> => {
    try {
      const sessionData = localStorage.getItem(`${finalConfig.storagePrefix}_${sessionId}`);
      if (!sessionData) return false;

      const { session: loadedSession, emotionHistory: loadedHistory, contextMemory: loadedMemory } = 
        JSON.parse(sessionData);

      // 恢复日期对象
      loadedSession.startTime = new Date(loadedSession.startTime);
      loadedSession.lastActivity = new Date(loadedSession.lastActivity);
      
      loadedHistory.forEach((emotion: any) => {
        emotion.timestamp = new Date(emotion.timestamp);
      });
      
      loadedMemory.forEach((memory: any) => {
        memory.timestamp = new Date(memory.timestamp);
      });

      setSession(loadedSession);
      setEmotionHistory(loadedHistory);
      setContextMemory(loadedMemory);
      sessionIdRef.current = sessionId;

      toast.success('会话已加载');
      return true;
    } catch (error) {
      logger.error('加载会话失败:', error);
      toast.error('加载会话失败');
      return false;
    }
  }, [finalConfig.storagePrefix]);

  // 清理会话
  const clearSession = useCallback(() => {
    setSession(null);
    setEmotionHistory([]);
    setContextMemory([]);
    setCurrentEmotion(null);
    sessionIdRef.current = null;
    
    if (finalConfig.autoSaveSession) {
      localStorage.removeItem(`${finalConfig.storagePrefix}_current`);
    }
    
    toast.info('会话已清理');
  }, [finalConfig.autoSaveSession, finalConfig.storagePrefix]);

  // WebSocket事件处理
  const setupWebSocketEvents = useCallback((ws: EmotionWebSocketService) => {
    ws.on('connect', () => {
      setError(null);
      toast.success('已连接到情感分析服务');
    });

    ws.on('disconnect', (reason) => {
      toast.info(`连接已断开: ${reason}`);
    });

    ws.on('emotion_result', (data: UnifiedEmotionalData) => {
      setCurrentEmotion(data);
      addEmotionToHistory(data);
      updateSessionActivity();
      
      // 更新统计
      setStats(ws.getConnectionStats());
    });

    ws.on('error', (error) => {
      const errorMsg = typeof error === 'string' ? error : error.message || '未知错误';
      setError(errorMsg);
      toast.error(`WebSocket错误: ${errorMsg}`);
    });

    ws.on('state_change', (state: ConnectionState) => {
      setConnectionState(state);
    });

    ws.on('heartbeat', () => {
      setStats(ws.getConnectionStats());
    });
  }, [addEmotionToHistory, updateSessionActivity]);

  // 连接WebSocket
  const connect = useCallback(async () => {
    if (wsServiceRef.current) {
      await wsServiceRef.current.connect();
      return;
    }

    const wsConfig = {
      ...finalConfig,
      userId: finalConfig.userId
    };

    const ws = createEmotionWebSocketService(wsConfig);
    wsServiceRef.current = ws;

    setupWebSocketEvents(ws);
    
    // 初始化或恢复会话
    if (!session) {
      initializeSession();
    }

    await ws.connect();
  }, [finalConfig, setupWebSocketEvents, session, initializeSession]);

  // 断开连接
  const disconnect = useCallback(() => {
    if (wsServiceRef.current) {
      wsServiceRef.current.disconnect();
      wsServiceRef.current = null;
    }
    
    // 自动保存会话
    if (finalConfig.autoSaveSession && session) {
      saveSession();
    }
  }, [finalConfig.autoSaveSession, session, saveSession]);

  // 发送方法封装
  const sendEmotionInput = useCallback(async (input: EmotionInputMessage) => {
    if (!wsServiceRef.current?.isConnected()) {
      throw new Error('WebSocket未连接');
    }
    
    await wsServiceRef.current.sendEmotionInput(input);
    updateSessionActivity();
  }, [updateSessionActivity]);

  const sendTextEmotion = useCallback(async (text: string) => {
    if (!wsServiceRef.current?.isConnected()) {
      throw new Error('WebSocket未连接');
    }
    
    await wsServiceRef.current.sendTextEmotion(text);
    updateSessionActivity();
    
    // 添加交互记忆
    addContextMemory({
      type: 'interaction',
      content: { type: 'text_input', text },
      importance: 0.6,
      tags: ['user_input', 'text']
    });
  }, [updateSessionActivity, addContextMemory]);

  const sendAudioEmotion = useCallback(async (audioBlob: Blob) => {
    if (!wsServiceRef.current?.isConnected()) {
      throw new Error('WebSocket未连接');
    }
    
    await wsServiceRef.current.sendAudioEmotion(audioBlob);
    updateSessionActivity();
    
    addContextMemory({
      type: 'interaction',
      content: { type: 'audio_input', size: audioBlob.size },
      importance: 0.7,
      tags: ['user_input', 'audio']
    });
  }, [updateSessionActivity, addContextMemory]);

  const sendVideoEmotion = useCallback(async (videoBlob: Blob) => {
    if (!wsServiceRef.current?.isConnected()) {
      throw new Error('WebSocket未连接');
    }
    
    await wsServiceRef.current.sendVideoEmotion(videoBlob);
    updateSessionActivity();
    
    addContextMemory({
      type: 'interaction',
      content: { type: 'video_input', size: videoBlob.size },
      importance: 0.8,
      tags: ['user_input', 'video']
    });
  }, [updateSessionActivity, addContextMemory]);

  const sendImageEmotion = useCallback(async (imageFile: File) => {
    if (!wsServiceRef.current?.isConnected()) {
      throw new Error('WebSocket未连接');
    }
    
    await wsServiceRef.current.sendImageEmotion(imageFile);
    updateSessionActivity();
    
    addContextMemory({
      type: 'interaction',
      content: { type: 'image_input', name: imageFile.name, size: imageFile.size },
      importance: 0.7,
      tags: ['user_input', 'image']
    });
  }, [updateSessionActivity, addContextMemory]);

  const sendPhysiologicalData = useCallback(async (data: Record<string, any>) => {
    if (!wsServiceRef.current?.isConnected()) {
      throw new Error('WebSocket未连接');
    }
    
    await wsServiceRef.current.sendPhysiologicalData(data);
    updateSessionActivity();
    
    addContextMemory({
      type: 'interaction',
      content: { type: 'physiological_input', data },
      importance: 0.9,
      tags: ['user_input', 'physiological']
    });
  }, [updateSessionActivity, addContextMemory]);

  // 清除错误
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // 组件挂载时的初始化
  useEffect(() => {
    // 启动记忆衰减定时器
    if (finalConfig.memoryDecayRate > 0) {
      memoryDecayTimerRef.current = setInterval(() => {
        processMemoryDecay();
      }, 60000); // 每分钟执行一次衰减
    }

    // 尝试恢复之前的会话
    if (finalConfig.autoSaveSession) {
      const currentSessionData = localStorage.getItem(`${finalConfig.storagePrefix}_current`);
      if (currentSessionData) {
        try {
          const { sessionId } = JSON.parse(currentSessionData);
          loadSession(sessionId);
        } catch (error) {
          logger.error('恢复会话失败:', error);
        }
      }
    }

    return () => {
      if (memoryDecayTimerRef.current) {
        clearInterval(memoryDecayTimerRef.current);
      }
    };
  }, [finalConfig.memoryDecayRate, finalConfig.autoSaveSession, finalConfig.storagePrefix, processMemoryDecay, loadSession]);

  // 组件卸载时清理
  useEffect(() => {
    return () => {
      if (wsServiceRef.current) {
        wsServiceRef.current.disconnect();
      }
      
      // 自动保存会话
      if (finalConfig.autoSaveSession && session) {
        saveSession();
      }
    };
  }, [finalConfig.autoSaveSession, session, saveSession]);

  return {
    // 连接状态
    connectionState,
    isConnected: connectionState === ConnectionState.CONNECTED,
    stats,
    
    // 会话管理
    session,
    sessionId: sessionIdRef.current,
    
    // 情感数据
    currentEmotion,
    emotionHistory,
    contextMemory,
    
    // 操作方法
    connect,
    disconnect,
    sendEmotionInput,
    sendTextEmotion,
    sendAudioEmotion,
    sendVideoEmotion,
    sendImageEmotion,
    sendPhysiologicalData,
    
    // 会话和记忆管理
    saveSession,
    loadSession,
    clearSession,
    addContextMemory,
    getRelevantMemories,
    updateMemoryImportance,
    
    // 错误处理
    error,
    clearError
  };
}
