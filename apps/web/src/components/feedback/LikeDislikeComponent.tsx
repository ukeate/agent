/**
 * 点赞/踩反馈组件
 * 支持点赞和踩的互斥操作
 */

import React, { useState, useCallback } from 'react'
import { feedbackService } from '../../services/feedbackService'

import { logger } from '../../utils/logger'
export interface LikeDislikeComponentProps {
  itemId?: string
  userId: string
  sessionId: string
  initialState?: 'none' | 'like' | 'dislike'
  size?: 'small' | 'medium' | 'large'
  disabled?: boolean
  showCounts?: boolean
  likesCount?: number
  dislikesCount?: number
  onStateChange?: (state: 'none' | 'like' | 'dislike') => void
  onSubmitSuccess?: (action: 'like' | 'dislike' | 'remove') => void
  onSubmitError?: (error: Error) => void
}

export const LikeDislikeComponent: React.FC<LikeDislikeComponentProps> = ({
  itemId,
  userId,
  sessionId,
  initialState = 'none',
  size = 'medium',
  disabled = false,
  showCounts = false,
  likesCount = 0,
  dislikesCount = 0,
  onStateChange,
  onSubmitSuccess,
  onSubmitError,
}) => {
  const [currentState, setCurrentState] = useState<'none' | 'like' | 'dislike'>(
    initialState
  )
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [counts, setCounts] = useState({
    likes: likesCount,
    dislikes: dislikesCount,
  })

  // 获取图标大小样式
  const getIconSize = () => {
    switch (size) {
      case 'small':
        return 'w-4 h-4'
      case 'large':
        return 'w-7 h-7'
      default:
        return 'w-5 h-5'
    }
  }

  // 获取按钮大小样式
  const getButtonSize = () => {
    switch (size) {
      case 'small':
        return 'p-1'
      case 'large':
        return 'p-3'
      default:
        return 'p-2'
    }
  }

  // 处理点赞/踩操作
  const handleAction = useCallback(
    async (action: 'like' | 'dislike') => {
      if (disabled || isSubmitting) return

      const previousState = currentState
      const newState = currentState === action ? 'none' : action

      // 乐观更新UI
      setCurrentState(newState)
      onStateChange?.(newState)

      // 更新计数（乐观更新）
      if (showCounts) {
        setCounts(prev => {
          const newCounts = { ...prev }

          // 撤销之前的状态
          if (previousState === 'like') newCounts.likes--
          if (previousState === 'dislike') newCounts.dislikes--

          // 应用新状态
          if (newState === 'like') newCounts.likes++
          if (newState === 'dislike') newCounts.dislikes++

          return newCounts
        })
      }

      try {
        setIsSubmitting(true)

        // 提交反馈
        const isLike = newState === 'like'
        const shouldSubmit = newState !== 'none'

        if (shouldSubmit) {
          await feedbackService.submitLike(
            userId,
            itemId || 'unknown',
            isLike,
            sessionId,
            {
              component: 'LikeDislikeComponent',
              action: newState,
              previousState,
              timestamp: Date.now(),
            }
          )
        }

        onSubmitSuccess?.(newState === 'none' ? 'remove' : newState)
      } catch (error) {
        logger.error('提交反馈失败:', error)

        // 回滚状态
        setCurrentState(previousState)
        onStateChange?.(previousState)

        // 回滚计数
        if (showCounts) {
          setCounts({ likes: likesCount, dislikes: dislikesCount })
        }

        onSubmitError?.(error as Error)
      } finally {
        setIsSubmitting(false)
      }
    },
    [
      disabled,
      isSubmitting,
      currentState,
      userId,
      itemId,
      sessionId,
      showCounts,
      likesCount,
      dislikesCount,
      onStateChange,
      onSubmitSuccess,
      onSubmitError,
    ]
  )

  // 点赞按钮样式
  const getLikeButtonStyle = () => {
    const baseStyle = `
      ${getButtonSize()}
      rounded-full transition-all duration-200 ease-in-out
      flex items-center justify-center space-x-1
      ${disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer hover:scale-105'}
      ${isSubmitting ? 'animate-pulse' : ''}
    `

    if (currentState === 'like') {
      return `${baseStyle} bg-blue-100 text-blue-600 ring-2 ring-blue-300 active`
    }
    return `${baseStyle} bg-gray-100 text-gray-600 hover:bg-blue-50 hover:text-blue-600`
  }

  // 踩按钮样式
  const getDislikeButtonStyle = () => {
    const baseStyle = `
      ${getButtonSize()}
      rounded-full transition-all duration-200 ease-in-out
      flex items-center justify-center space-x-1
      ${disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer hover:scale-105'}
      ${isSubmitting ? 'animate-pulse' : ''}
    `

    if (currentState === 'dislike') {
      return `${baseStyle} bg-red-100 text-red-600 ring-2 ring-red-300 active`
    }
    return `${baseStyle} bg-gray-100 text-gray-600 hover:bg-red-50 hover:text-red-600`
  }

  return (
    <div className="flex items-center space-x-2">
      {/* 点赞按钮 */}
      <button
        type="button"
        className={getLikeButtonStyle()}
        disabled={disabled || isSubmitting}
        onClick={() => handleAction('like')}
        aria-label={currentState === 'like' ? '取消点赞' : '点赞'}
        title={currentState === 'like' ? '取消点赞' : '点赞'}
        data-testid="like-button"
      >
        <svg
          className={`${getIconSize()} transition-transform duration-200 ${
            currentState === 'like' ? 'scale-110' : ''
          }`}
          fill={currentState === 'like' ? 'currentColor' : 'none'}
          stroke="currentColor"
          strokeWidth={2}
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5"
          />
        </svg>

        {showCounts && (
          <span
            className={`text-xs font-medium ${size === 'small' ? 'hidden' : ''}`}
          >
            {counts.likes}
          </span>
        )}
      </button>

      {/* 踩按钮 */}
      <button
        type="button"
        className={getDislikeButtonStyle()}
        disabled={disabled || isSubmitting}
        onClick={() => handleAction('dislike')}
        aria-label={currentState === 'dislike' ? '取消踩' : '踩'}
        title={currentState === 'dislike' ? '取消踩' : '踩'}
        data-testid="dislike-button"
      >
        <svg
          className={`${getIconSize()} transition-transform duration-200 transform rotate-180 ${
            currentState === 'dislike' ? 'scale-110' : ''
          }`}
          fill={currentState === 'dislike' ? 'currentColor' : 'none'}
          stroke="currentColor"
          strokeWidth={2}
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5"
          />
        </svg>

        {showCounts && (
          <span
            className={`text-xs font-medium ${size === 'small' ? 'hidden' : ''}`}
          >
            {counts.dislikes}
          </span>
        )}
      </button>

      {/* 状态指示器 */}
      {isSubmitting && (
        <div className="flex items-center space-x-1">
          <svg
            className="animate-spin h-3 w-3 text-gray-400"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
              fill="none"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
          {size !== 'small' && (
            <span className="text-xs text-gray-500">提交中...</span>
          )}
        </div>
      )}
    </div>
  )
}

export default LikeDislikeComponent
