import React, { useState, useEffect, useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Button } from '@/components/ui/button'
import {
  Heart,
  Brain,
  Activity,
  TrendingUp,
  TrendingDown,
  Minus,
  Sparkles,
  Timer,
  Users,
  AlertTriangle,
  CheckCircle,
  Info,
} from 'lucide-react'
import { cn } from '@/lib/utils'

// 情感状态定义
export interface EmotionState {
  emotion: EmotionType
  intensity: number
  valence: number
  arousal: number
  dominance: number
  confidence: number
  timestamp: Date
}

export enum EmotionType {
  HAPPINESS = 'happiness',
  SADNESS = 'sadness',
  ANGER = 'anger',
  FEAR = 'fear',
  SURPRISE = 'surprise',
  DISGUST = 'disgust',
  NEUTRAL = 'neutral',
}

// 多模态情感结果
export interface MultiModalEmotion {
  emotions: Record<string, EmotionState>
  fusedEmotion: EmotionState
  confidence: number
  processingTime: number
}

// 个性特征
export interface PersonalityProfile {
  openness: number
  conscientiousness: number
  extraversion: number
  agreeableness: number
  neuroticism: number
  updatedAt: Date
}

// 共情响应
export interface EmpathyResponse {
  message: string
  responseType: string
  confidence: number
  generationStrategy: string
}

// 统一情感数据
export interface UnifiedEmotionalData {
  userId: string
  timestamp: Date
  recognitionResult?: MultiModalEmotion
  emotionalState?: EmotionState
  personalityProfile?: PersonalityProfile
  empathyResponse?: EmpathyResponse
  confidence: number
  processingTime: number
  dataQuality: number
}

interface EmotionalFeedbackDisplayProps {
  emotionalData?: UnifiedEmotionalData
  isRealTime?: boolean
  showHistory?: boolean
  showPersonality?: boolean
  showEmpathy?: boolean
  className?: string
}

// 情感颜色映射
const EMOTION_COLORS = {
  [EmotionType.HAPPINESS]: 'text-yellow-600 bg-yellow-50 border-yellow-200',
  [EmotionType.SADNESS]: 'text-blue-600 bg-blue-50 border-blue-200',
  [EmotionType.ANGER]: 'text-red-600 bg-red-50 border-red-200',
  [EmotionType.FEAR]: 'text-purple-600 bg-purple-50 border-purple-200',
  [EmotionType.SURPRISE]: 'text-orange-600 bg-orange-50 border-orange-200',
  [EmotionType.DISGUST]: 'text-green-600 bg-green-50 border-green-200',
  [EmotionType.NEUTRAL]: 'text-gray-600 bg-gray-50 border-gray-200',
}

// 情感图标映射
const EMOTION_ICONS = {
  [EmotionType.HAPPINESS]: '😊',
  [EmotionType.SADNESS]: '😢',
  [EmotionType.ANGER]: '😠',
  [EmotionType.FEAR]: '😨',
  [EmotionType.SURPRISE]: '😲',
  [EmotionType.DISGUST]: '🤢',
  [EmotionType.NEUTRAL]: '😐',
}

// 情感名称映射
const EMOTION_NAMES = {
  [EmotionType.HAPPINESS]: '快乐',
  [EmotionType.SADNESS]: '悲伤',
  [EmotionType.ANGER]: '愤怒',
  [EmotionType.FEAR]: '恐惧',
  [EmotionType.SURPRISE]: '惊讶',
  [EmotionType.DISGUST]: '厌恶',
  [EmotionType.NEUTRAL]: '中性',
}

// VAD维度组件
const VADDisplay: React.FC<{ emotion: EmotionState }> = ({ emotion }) => {
  const getValenceLabel = (valence: number) => {
    if (valence > 0.3)
      return { label: '积极', color: 'text-green-600', icon: TrendingUp }
    if (valence < -0.3)
      return { label: '消极', color: 'text-red-600', icon: TrendingDown }
    return { label: '中性', color: 'text-gray-600', icon: Minus }
  }

  const getArousalLabel = (arousal: number) => {
    if (arousal > 0.6) return '高激活'
    if (arousal > 0.4) return '中激活'
    return '低激活'
  }

  const getDominanceLabel = (dominance: number) => {
    if (dominance > 0.6) return '主导'
    if (dominance > 0.4) return '平衡'
    return '顺从'
  }

  const valenceInfo = getValenceLabel(emotion.valence)
  const ValenceIcon = valenceInfo.icon

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-3 gap-4">
        {/* 效价 */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <ValenceIcon className={cn('h-4 w-4', valenceInfo.color)} />
            <span className="text-sm font-medium">效价</span>
          </div>
          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <span>消极</span>
              <span className={valenceInfo.color}>{valenceInfo.label}</span>
              <span>积极</span>
            </div>
            <Progress
              value={((emotion.valence + 1) / 2) * 100}
              className="h-2"
            />
            <div className="text-xs text-center text-muted-foreground">
              {emotion.valence.toFixed(2)}
            </div>
          </div>
        </div>

        {/* 激活度 */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-blue-600" />
            <span className="text-sm font-medium">激活</span>
          </div>
          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <span>低</span>
              <span className="text-blue-600">
                {getArousalLabel(emotion.arousal)}
              </span>
              <span>高</span>
            </div>
            <Progress value={emotion.arousal * 100} className="h-2" />
            <div className="text-xs text-center text-muted-foreground">
              {emotion.arousal.toFixed(2)}
            </div>
          </div>
        </div>

        {/* 优势度 */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Users className="h-4 w-4 text-purple-600" />
            <span className="text-sm font-medium">优势</span>
          </div>
          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <span>顺从</span>
              <span className="text-purple-600">
                {getDominanceLabel(emotion.dominance)}
              </span>
              <span>主导</span>
            </div>
            <Progress value={emotion.dominance * 100} className="h-2" />
            <div className="text-xs text-center text-muted-foreground">
              {emotion.dominance.toFixed(2)}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// 置信度指示器
const ConfidenceIndicator: React.FC<{
  confidence: number
  size?: 'sm' | 'md' | 'lg'
}> = ({ confidence, size = 'md' }) => {
  const getConfidenceLevel = (conf: number) => {
    if (conf >= 0.8)
      return { level: '高', color: 'text-green-600', icon: CheckCircle }
    if (conf >= 0.6)
      return { level: '中', color: 'text-yellow-600', icon: Info }
    return { level: '低', color: 'text-red-600', icon: AlertTriangle }
  }

  const confInfo = getConfidenceLevel(confidence)
  const ConfIcon = confInfo.icon

  const sizeClasses = {
    sm: 'h-3 w-3',
    md: 'h-4 w-4',
    lg: 'h-5 w-5',
  }

  return (
    <div className="flex items-center gap-1">
      <ConfIcon className={cn(sizeClasses[size], confInfo.color)} />
      <span className={cn('text-xs font-medium', confInfo.color)}>
        {confInfo.level}置信度
      </span>
      <span className="text-xs text-muted-foreground">
        ({Math.round(confidence * 100)}%)
      </span>
    </div>
  )
}

// 个性特征雷达图（简化版）
const PersonalityRadar: React.FC<{ personality: PersonalityProfile }> = ({
  personality,
}) => {
  const traits = [
    { key: 'openness', label: '开放性', value: personality.openness },
    {
      key: 'conscientiousness',
      label: '责任心',
      value: personality.conscientiousness,
    },
    { key: 'extraversion', label: '外向性', value: personality.extraversion },
    { key: 'agreeableness', label: '宜人性', value: personality.agreeableness },
    { key: 'neuroticism', label: '神经质', value: personality.neuroticism },
  ]

  return (
    <div className="space-y-3">
      {traits.map(trait => (
        <div key={trait.key} className="space-y-1">
          <div className="flex justify-between text-sm">
            <span>{trait.label}</span>
            <span className="font-mono text-xs">
              {(trait.value * 100).toFixed(0)}%
            </span>
          </div>
          <Progress value={trait.value * 100} className="h-2" />
        </div>
      ))}
    </div>
  )
}

// 主组件
export const EmotionalFeedbackDisplay: React.FC<
  EmotionalFeedbackDisplayProps
> = ({
  emotionalData,
  isRealTime = false,
  showHistory = false,
  showPersonality = true,
  showEmpathy = true,
  className,
}) => {
  const [animationKey, setAnimationKey] = useState(0)
  const [emotionHistory, setEmotionHistory] = useState<EmotionState[]>([])

  // 当收到新数据时触发动画
  useEffect(() => {
    if (emotionalData) {
      setAnimationKey(prev => prev + 1)

      if (emotionalData.emotionalState && showHistory) {
        setEmotionHistory(prev => {
          const newHistory = [...prev, emotionalData.emotionalState!]
          return newHistory.slice(-10) // 保留最近10条记录
        })
      }
    }
  }, [emotionalData, showHistory])

  // 主要情感状态
  const primaryEmotion = useMemo(() => {
    return (
      emotionalData?.emotionalState ||
      emotionalData?.recognitionResult?.fusedEmotion
    )
  }, [emotionalData])

  // 多模态情感结果
  const modalityEmotions = useMemo(() => {
    return emotionalData?.recognitionResult?.emotions || {}
  }, [emotionalData])

  if (!emotionalData) {
    return (
      <Card className={cn('w-full', className)}>
        <CardContent className="flex items-center justify-center h-48 text-muted-foreground">
          <div className="text-center space-y-2">
            <Brain className="h-8 w-8 mx-auto opacity-50" />
            <p>等待情感数据...</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className={cn('w-full', className)}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5" />
            情感分析结果
            {isRealTime && (
              <Badge variant="outline" className="text-green-600">
                <Activity className="h-3 w-3 mr-1" />
                实时
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Timer className="h-4 w-4" />
            {emotionalData.processingTime.toFixed(2)}s
          </div>
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* 主要情感状态 */}
        {primaryEmotion && (
          <div
            key={`primary-${animationKey}`}
            className={cn(
              'p-4 rounded-lg border-2 transition-all duration-500 animate-in fade-in-0 slide-in-from-bottom-2',
              EMOTION_COLORS[primaryEmotion.emotion]
            )}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-3">
                <span className="text-3xl">
                  {EMOTION_ICONS[primaryEmotion.emotion]}
                </span>
                <div>
                  <h3 className="text-lg font-semibold">
                    {EMOTION_NAMES[primaryEmotion.emotion]}
                  </h3>
                  <div className="flex items-center gap-2">
                    <span className="text-sm opacity-80">
                      强度: {Math.round(primaryEmotion.intensity * 100)}%
                    </span>
                    <ConfidenceIndicator
                      confidence={primaryEmotion.confidence}
                      size="sm"
                    />
                  </div>
                </div>
              </div>
              <div className="text-right">
                <Progress
                  value={primaryEmotion.intensity * 100}
                  className="w-20 h-2 mb-1"
                />
              </div>
            </div>

            {/* VAD维度显示 */}
            <VADDisplay emotion={primaryEmotion} />
          </div>
        )}

        {/* 多模态结果 */}
        {Object.keys(modalityEmotions).length > 0 && (
          <div className="space-y-3">
            <h4 className="text-sm font-medium text-muted-foreground">
              各模态情感识别结果
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {Object.entries(modalityEmotions).map(([modality, emotion]) => (
                <div key={modality} className="p-3 bg-muted rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary" className="text-xs">
                        {modality.toUpperCase()}
                      </Badge>
                      <span className="text-sm font-medium">
                        {EMOTION_NAMES[emotion.emotion]}
                      </span>
                    </div>
                    <span className="text-lg">
                      {EMOTION_ICONS[emotion.emotion]}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Progress
                      value={emotion.intensity * 100}
                      className="flex-1 h-1"
                    />
                    <span className="text-xs text-muted-foreground">
                      {Math.round(emotion.confidence * 100)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 个性特征 */}
        {showPersonality && emotionalData.personalityProfile && (
          <div className="space-y-3">
            <h4 className="text-sm font-medium text-muted-foreground">
              个性特征分析
            </h4>
            <div className="p-4 bg-muted rounded-lg">
              <PersonalityRadar
                personality={emotionalData.personalityProfile}
              />
            </div>
          </div>
        )}

        {/* 共情响应 */}
        {showEmpathy && emotionalData.empathyResponse && (
          <div className="space-y-3">
            <h4 className="text-sm font-medium text-muted-foreground">
              智能回应建议
            </h4>
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="flex items-start gap-3">
                <Heart className="h-5 w-5 text-blue-600 mt-0.5" />
                <div className="flex-1">
                  <p className="text-sm leading-relaxed mb-2">
                    {emotionalData.empathyResponse.message}
                  </p>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-xs">
                      {emotionalData.empathyResponse.responseType}
                    </Badge>
                    <ConfidenceIndicator
                      confidence={emotionalData.empathyResponse.confidence}
                      size="sm"
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 数据质量和系统信息 */}
        <div className="pt-3 border-t border-muted">
          <div className="grid grid-cols-2 gap-4 text-xs text-muted-foreground">
            <div>
              <span>数据质量: </span>
              <span className="font-mono">
                {Math.round(emotionalData.dataQuality * 100)}%
              </span>
            </div>
            <div>
              <span>总置信度: </span>
              <span className="font-mono">
                {Math.round(emotionalData.confidence * 100)}%
              </span>
            </div>
          </div>

          {emotionalData.timestamp && (
            <div className="text-xs text-muted-foreground mt-2">
              分析时间: {emotionalData.timestamp.toLocaleString()}
            </div>
          )}
        </div>

        {/* 情感历史趋势 */}
        {showHistory && emotionHistory.length > 1 && (
          <div className="space-y-3">
            <h4 className="text-sm font-medium text-muted-foreground">
              情感变化趋势
            </h4>
            <div className="flex items-center gap-1 p-2 bg-muted rounded-lg overflow-x-auto">
              {emotionHistory.map((emotion, index) => (
                <div
                  key={index}
                  className="flex flex-col items-center gap-1 min-w-0 flex-shrink-0"
                >
                  <span className="text-sm">
                    {EMOTION_ICONS[emotion.emotion]}
                  </span>
                  <div className="w-1 h-8 bg-muted-foreground/20 rounded">
                    <div
                      className="w-full bg-blue-500 rounded"
                      style={{ height: `${emotion.intensity * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {index + 1}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
