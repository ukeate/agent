import { buildWsUrl } from '../utils/apiBase'
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import {
import { logger } from '../utils/logger'
  Heart,
  Brain,
  Activity,
  Wifi,
  WifiOff,
  Settings,
  History,
  MemoryStick,
  AlertCircle,
  CheckCircle,
  Loader2
} from 'lucide-react';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';

import { EmotionalInputPanel, EmotionalInputData, ModalityType } from '@/components/emotion/EmotionalInputPanel';
import { EmotionalFeedbackDisplay } from '@/components/emotion/EmotionalFeedbackDisplay';
import { Emotion3DVisualizer, EmotionState } from '@/components/emotion/Emotion3DVisualizer';
import { useEmotionWebSocket, ContextMemory } from '@/hooks/useEmotionWebSocket';
import { ConnectionState } from '@/services/emotionWebSocketService';

// 连接状态指示器组件
const ConnectionStatusIndicator: React.FC<{
  state: ConnectionState;
  stats: any;
}> = ({ state, stats }) => {
  const getStatusInfo = (state: ConnectionState) => {
    switch (state) {
      case ConnectionState.CONNECTED:
        return {
          icon: Wifi,
          label: '已连接',
          color: 'text-green-600 bg-green-50 border-green-200',
          variant: 'default' as const
        };
      case ConnectionState.CONNECTING:
        return {
          icon: Loader2,
          label: '连接中',
          color: 'text-blue-600 bg-blue-50 border-blue-200',
          variant: 'secondary' as const
        };
      case ConnectionState.RECONNECTING:
        return {
          icon: Loader2,
          label: '重连中',
          color: 'text-yellow-600 bg-yellow-50 border-yellow-200',
          variant: 'outline' as const
        };
      case ConnectionState.ERROR:
        return {
          icon: AlertCircle,
          label: '连接错误',
          color: 'text-red-600 bg-red-50 border-red-200',
          variant: 'destructive' as const
        };
      default:
        return {
          icon: WifiOff,
          label: '未连接',
          color: 'text-gray-600 bg-gray-50 border-gray-200',
          variant: 'outline' as const
        };
    }
  };

  const statusInfo = getStatusInfo(state);
  const StatusIcon = statusInfo.icon;
  const isLoading = state === ConnectionState.CONNECTING || state === ConnectionState.RECONNECTING;

  return (
    <div className={cn("p-3 rounded-lg border", statusInfo.color)}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <StatusIcon className={cn("h-4 w-4", isLoading && "animate-spin")} />
          <span className="text-sm font-medium">{statusInfo.label}</span>
        </div>
        {stats && (
          <div className="text-xs opacity-70">
            发送: {stats.messagesSent} | 接收: {stats.messagesReceived}
          </div>
        )}
      </div>
      
      {state === ConnectionState.CONNECTED && stats && (
        <div className="mt-2 text-xs opacity-70">
          <div>连接时间: {stats.connectionTime ? new Date(stats.connectionTime).toLocaleTimeString() : '-'}</div>
          {stats.lastHeartbeat && (
            <div>最后心跳: {new Date(stats.lastHeartbeat).toLocaleTimeString()}</div>
          )}
        </div>
      )}
    </div>
  );
};

// 会话信息组件
const SessionInfoPanel: React.FC<{
  session: any;
  emotionHistory: any[];
  contextMemory: ContextMemory[];
  onClearSession: () => void;
  onSaveSession: () => void;
}> = ({ session, emotionHistory, contextMemory, onClearSession, onSaveSession }) => {
  if (!session) return null;

  const sessionDuration = session.startTime ? 
    Math.floor((Date.now() - new Date(session.startTime).getTime()) / 60000) : 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm">
          <Activity className="h-4 w-4" />
          会话信息
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <div className="text-muted-foreground">会话ID</div>
            <div className="font-mono text-xs">{session.sessionId.substring(0, 12)}...</div>
          </div>
          <div>
            <div className="text-muted-foreground">用户ID</div>
            <div className="font-mono text-xs">{session.userId}</div>
          </div>
          <div>
            <div className="text-muted-foreground">会话时长</div>
            <div>{sessionDuration} 分钟</div>
          </div>
          <div>
            <div className="text-muted-foreground">消息数量</div>
            <div>{session.messageCount}</div>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span>情感历史</span>
            <Badge variant="outline">{emotionHistory.length}</Badge>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span>上下文记忆</span>
            <Badge variant="outline">{contextMemory.length}</Badge>
          </div>
        </div>

        <div className="flex gap-2">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={onSaveSession}
            className="flex-1"
          >
            保存会话
          </Button>
          <Button 
            variant="outline" 
            size="sm" 
            onClick={onClearSession}
            className="flex-1"
          >
            清空会话
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

// 记忆浏览器组件
const MemoryBrowser: React.FC<{
  memories: ContextMemory[];
  onUpdateImportance: (memoryId: string, importance: number) => void;
}> = ({ memories, onUpdateImportance }) => {
  const [selectedType, setSelectedType] = useState<string>('all');
  
  const filteredMemories = memories.filter(memory => 
    selectedType === 'all' || memory.type === selectedType
  );

  const memoryTypes = ['all', ...new Set(memories.map(m => m.type))];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm">
          <MemoryStick className="h-4 w-4" />
          上下文记忆
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex gap-2 flex-wrap">
          {memoryTypes.map(type => (
            <Button
              key={type}
              variant={selectedType === type ? "default" : "outline"}
              size="sm"
              onClick={() => setSelectedType(type)}
              className="text-xs"
            >
              {type === 'all' ? '全部' : type}
            </Button>
          ))}
        </div>

        <div className="space-y-2 max-h-60 overflow-y-auto">
          {filteredMemories.length === 0 ? (
            <div className="text-center text-muted-foreground text-sm py-4">
              暂无记忆数据
            </div>
          ) : (
            filteredMemories.map(memory => (
              <div key={memory.id} className="p-2 bg-muted rounded-lg">
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <Badge variant="secondary" className="text-xs">
                        {memory.type}
                      </Badge>
                      {memory.tags?.map(tag => (
                        <Badge key={tag} variant="outline" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                    
                    <div className="text-xs text-muted-foreground mb-2">
                      {memory.timestamp.toLocaleString()}
                    </div>

                    <div className="text-xs">
                      {typeof memory.content === 'string' 
                        ? memory.content 
                        : JSON.stringify(memory.content).substring(0, 100) + '...'
                      }
                    </div>
                  </div>
                </div>

                <div className="mt-2 space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span>重要性</span>
                    <span>{Math.round(memory.importance * 100)}%</span>
                  </div>
                  <Progress value={memory.importance * 100} className="h-1" />
                </div>

                {memory.decay !== undefined && (
                  <div className="mt-1">
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>衰减</span>
                      <span>{Math.round(memory.decay * 100)}%</span>
                    </div>
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
};

// 转换情感数据格式为3D可视化格式
const convertToEmotionState = (emotionData: any): EmotionState | undefined => {
  if (!emotionData) return undefined;
  
  return {
    emotion: emotionData.primary_emotion || 'neutral',
    intensity: emotionData.intensity || 0.5,
    valence: emotionData.valence || 0,
    arousal: emotionData.arousal || 0.5,
    dominance: emotionData.dominance || 0,
    confidence: emotionData.confidence || 0.8,
    timestamp: new Date(emotionData.timestamp || Date.now())
  };
};

// 主页面组件
export default function EmotionInteractionPage() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState('interaction');

  // 使用WebSocket Hook
  const {
    connectionState,
    isConnected,
    stats,
    session,
    sessionId,
    currentEmotion,
    emotionHistory,
    contextMemory,
    connect,
    disconnect,
    sendTextEmotion,
    sendAudioEmotion,
    sendVideoEmotion,
    sendImageEmotion,
    sendPhysiologicalData,
    saveSession,
    clearSession,
    addContextMemory,
    getRelevantMemories,
    updateMemoryImportance,
    error,
    clearError
  } = useEmotionWebSocket({
    url: buildWsUrl('/ws'),
    userId: 'demo_user', // 在实际应用中应该从认证系统获取
    maxHistoryLength: 50,
    maxMemoryLength: 200,
    memoryDecayRate: 0.005,
    autoSaveSession: true
  });

  // 自动连接
  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  // 处理情感输入
  const handleEmotionSubmit = async (data: EmotionalInputData) => {
    setIsProcessing(true);
    
    try {
      // 根据模态类型分别发送
      for (const modality of data.modalities) {
        switch (modality) {
          case ModalityType.TEXT:
            if (data.text) {
              await sendTextEmotion(data.text);
            }
            break;
          case ModalityType.AUDIO:
            if (data.audioBlob) {
              await sendAudioEmotion(data.audioBlob);
            }
            break;
          case ModalityType.VIDEO:
            if (data.videoBlob) {
              await sendVideoEmotion(data.videoBlob);
            }
            break;
          case ModalityType.IMAGE:
            if (data.imageFile) {
              await sendImageEmotion(data.imageFile);
            }
            break;
          case ModalityType.PHYSIOLOGICAL:
            if (data.physiologicalData) {
              await sendPhysiologicalData(data.physiologicalData);
            }
            break;
        }
      }
      
      toast.success('情感数据已发送');
    } catch (error: any) {
      toast.error('发送失败: ' + error.message);
    } finally {
      setIsProcessing(false);
    }
  };

  // 处理实时数据
  const handleRealTimeData = (data: Partial<EmotionalInputData>) => {
    // 这里可以实现实时数据的预处理或预览
    logger.log('实时数据:', data);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* 页面标题 */}
      <div className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Brain className="h-8 w-8 text-primary" />
              <div>
                <h1 className="text-2xl font-bold">情感智能交互系统</h1>
                <p className="text-sm text-muted-foreground">
                  多模态情感分析与实时反馈平台
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <ConnectionStatusIndicator state={connectionState} stats={stats} />
              <Button
                variant={isConnected ? "outline" : "default"}
                onClick={isConnected ? disconnect : connect}
                disabled={connectionState === ConnectionState.CONNECTING}
              >
                {isConnected ? '断开连接' : '连接'}
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* 错误提示 */}
      {error && (
        <div className="container mx-auto px-4 py-2">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="flex items-center justify-between">
              <span>{error}</span>
              <Button variant="ghost" size="sm" onClick={clearError}>
                关闭
              </Button>
            </AlertDescription>
          </Alert>
        </div>
      )}

      {/* 主内容区域 */}
      <div className="container mx-auto px-4 py-6">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="interaction">情感交互</TabsTrigger>
            <TabsTrigger value="visualization">3D可视化</TabsTrigger>
            <TabsTrigger value="history">历史记录</TabsTrigger>
            <TabsTrigger value="memory">上下文记忆</TabsTrigger>
            <TabsTrigger value="session">会话管理</TabsTrigger>
          </TabsList>

          {/* 情感交互标签页 */}
          <TabsContent value="interaction" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* 输入面板 */}
              <div className="space-y-4">
                <EmotionalInputPanel
                  onSubmit={handleEmotionSubmit}
                  onRealTimeData={handleRealTimeData}
                  isProcessing={isProcessing}
                  enabledModalities={[
                    ModalityType.TEXT,
                    ModalityType.AUDIO,
                    ModalityType.VIDEO,
                    ModalityType.IMAGE,
                    ModalityType.PHYSIOLOGICAL
                  ]}
                />
              </div>

              {/* 反馈显示面板 */}
              <div className="space-y-4">
                <EmotionalFeedbackDisplay
                  emotionalData={currentEmotion}
                  isRealTime={true}
                  showHistory={false}
                  showPersonality={true}
                  showEmpathy={true}
                />
              </div>
            </div>
          </TabsContent>

          {/* 3D可视化标签页 */}
          <TabsContent value="visualization" className="space-y-6">
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
              {/* 3D情感空间可视化 */}
              <div className="xl:col-span-2">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Brain className="h-5 w-5" />
                      3D情感空间可视化
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Emotion3DVisualizer
                      emotionalState={convertToEmotionState(currentEmotion)}
                      history={emotionHistory.map(emotion => convertToEmotionState(emotion)).filter(Boolean) as EmotionState[]}
                      width={800}
                      height={500}
                      showControls={true}
                      showAxisLabels={true}
                      showTrajectory={true}
                      interactive={true}
                      onEmotionClick={(emotion) => {
                        logger.log('点击的情感:', emotion);
                        toast.info(`点击了情感: ${emotion.emotion} (强度: ${(emotion.intensity * 100).toFixed(1)}%)`);
                      }}
                    />
                  </CardContent>
                </Card>
              </div>

              {/* 情感分析面板 */}
              <div className="space-y-4">
                {currentEmotion && (
                  <EmotionalFeedbackDisplay
                    emotionalData={currentEmotion}
                    isRealTime={true}
                    showHistory={false}
                    showPersonality={true}
                    showEmpathy={true}
                  />
                )}

                {/* 实时统计 */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">实时统计</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <div className="text-muted-foreground">总情感点</div>
                        <div className="font-semibold">{emotionHistory.length}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">当前情感</div>
                        <div className="font-semibold">
                          {currentEmotion?.primary_emotion || '无'}
                        </div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">平均强度</div>
                        <div className="font-semibold">
                          {emotionHistory.length > 0
                            ? ((emotionHistory.reduce((sum, e) => sum + (e.intensity || 0), 0) / emotionHistory.length) * 100).toFixed(1) + '%'
                            : '0%'
                          }
                        </div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">会话时长</div>
                        <div className="font-semibold">
                          {session?.startTime 
                            ? Math.floor((Date.now() - new Date(session.startTime).getTime()) / 60000) + ' 分钟'
                            : '0 分钟'
                          }
                        </div>
                      </div>
                    </div>

                    {/* 情感分布 */}
                    {emotionHistory.length > 0 && (
                      <div className="space-y-2">
                        <div className="text-sm font-medium">情感分布</div>
                        {(() => {
                          const emotionCounts = emotionHistory.reduce((acc: Record<string, number>, emotion) => {
                            const key = emotion.primary_emotion || 'neutral';
                            acc[key] = (acc[key] || 0) + 1;
                            return acc;
                          }, {});
                          
                          const total = emotionHistory.length;
                          
                          return Object.entries(emotionCounts)
                            .sort(([,a], [,b]) => b - a)
                            .slice(0, 5)
                            .map(([emotion, count]) => (
                              <div key={emotion} className="flex items-center justify-between text-xs">
                                <span className="capitalize">{emotion}</span>
                                <div className="flex items-center gap-2">
                                  <div className="w-16 h-2 bg-muted rounded-full overflow-hidden">
                                    <div 
                                      className="h-full bg-primary rounded-full"
                                      style={{ width: `${(count / total) * 100}%` }}
                                    />
                                  </div>
                                  <span className="w-8 text-right">{count}</span>
                                </div>
                              </div>
                            ));
                        })()}
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* 历史记录标签页 */}
          <TabsContent value="history" className="space-y-4">
            {emotionHistory.length === 0 ? (
              <Card>
                <CardContent className="flex items-center justify-center h-48 text-muted-foreground">
                  <div className="text-center space-y-2">
                    <History className="h-8 w-8 mx-auto opacity-50" />
                    <p>暂无历史记录</p>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <div className="space-y-4">
                {emotionHistory.map((emotion, index) => (
                  <EmotionalFeedbackDisplay
                    key={`${emotion.user_id}-${emotion.timestamp}-${index}`}
                    emotionalData={emotion}
                    isRealTime={false}
                    showHistory={false}
                    showPersonality={false}
                    showEmpathy={true}
                  />
                ))}
              </div>
            )}
          </TabsContent>

          {/* 上下文记忆标签页 */}
          <TabsContent value="memory">
            <MemoryBrowser
              memories={contextMemory}
              onUpdateImportance={updateMemoryImportance}
            />
          </TabsContent>

          {/* 会话管理标签页 */}
          <TabsContent value="session">
            <SessionInfoPanel
              session={session}
              emotionHistory={emotionHistory}
              contextMemory={contextMemory}
              onClearSession={clearSession}
              onSaveSession={saveSession}
            />
          </TabsContent>
        </Tabs>
      </div>

      {/* 底部状态栏 */}
      {session && (
        <div className="fixed bottom-0 left-0 right-0 bg-background/95 backdrop-blur border-t p-2">
          <div className="container mx-auto flex items-center justify-between text-xs text-muted-foreground">
            <div className="flex items-center gap-4">
              <span>会话: {session.sessionId.substring(0, 8)}...</span>
              <span>消息: {session.messageCount}</span>
              <span>历史: {emotionHistory.length}</span>
              <span>记忆: {contextMemory.length}</span>
            </div>
            <div className="flex items-center gap-2">
              {isConnected ? (
                <CheckCircle className="h-3 w-3 text-green-500" />
              ) : (
                <AlertCircle className="h-3 w-3 text-red-500" />
              )}
              <span>
                {isConnected ? '在线' : '离线'}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
