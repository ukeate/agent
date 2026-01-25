/**
 * 评分反馈组件
 * 支持1-5星评分，包含动画效果和样式
 */

import React, { useState, useCallback } from 'react'
import { feedbackService } from '../../services/feedbackService'

import { logger } from '../../utils/logger'
export interface RatingComponentProps {
  itemId?: string
  userId: string
  sessionId: string
  initialRating?: number
  size?: 'small' | 'medium' | 'large'
  disabled?: boolean
  showLabel?: boolean
  onRatingChange?: (rating: number) => void
  onSubmitSuccess?: (rating: number) => void
  onSubmitError?: (error: Error) => void
}

const RATING_LABELS = ['极差', '较差', '一般', '较好', '极好']

export const RatingComponent: React.FC<RatingComponentProps> = ({
  itemId,
  userId,
  sessionId,
  initialRating = 0,
  size = 'medium',
  disabled = false,
  showLabel = true,
  onRatingChange,
  onSubmitSuccess,
  onSubmitError,
}) => {
  const [currentRating, setCurrentRating] = useState(initialRating)
  const [hoverRating, setHoverRating] = useState(0)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [hasSubmitted, setHasSubmitted] = useState(false)

  // 获取星星大小样式
  const getStarSize = () => {
    switch (size) {
      case 'small':
        return 'w-4 h-4'
      case 'large':
        return 'w-11 h-11'
      default:
        return 'w-6 h-6'
    }
  }

  // 处理评分点击
  const handleRatingClick = useCallback(
    async (rating: number) => {
      if (disabled || isSubmitting) return

      setCurrentRating(rating)
      onRatingChange?.(rating)

      // 提交评分
      try {
        setIsSubmitting(true)

        await feedbackService.submitRating(
          userId,
          itemId || 'unknown',
          rating,
          sessionId,
          {
            component: 'RatingComponent',
            timestamp: Date.now(),
            previousRating: currentRating,
          }
        )

        setHasSubmitted(true)
        onSubmitSuccess?.(rating)
      } catch (error) {
        logger.error('提交评分失败:', error)
        onSubmitError?.(error as Error)
        // 回滚评分
        setCurrentRating(initialRating)
      } finally {
        setIsSubmitting(false)
      }
    },
    [
      disabled,
      isSubmitting,
      userId,
      itemId,
      sessionId,
      currentRating,
      initialRating,
      onRatingChange,
      onSubmitSuccess,
      onSubmitError,
    ]
  )

  // 处理鼠标悬停
  const handleMouseEnter = useCallback(
    (rating: number) => {
      if (!disabled) {
        setHoverRating(rating)
      }
    },
    [disabled]
  )

  // 处理鼠标离开
  const handleMouseLeave = useCallback(() => {
    if (!disabled) {
      setHoverRating(0)
    }
  }, [disabled])

  // 获取显示的评分（悬停优先于当前评分）
  const displayRating = hoverRating || currentRating

  return (
    <div className="flex flex-col items-start space-y-2">
      {/* 星星评分 */}
      <div
        className="flex items-center space-x-1"
        onMouseLeave={handleMouseLeave}
      >
        {[1, 2, 3, 4, 5].map(star => (
          <button
            key={star}
            type="button"
            className={`
              ${getStarSize()}
              transition-all duration-200 ease-in-out
              ${
                disabled
                  ? 'cursor-not-allowed opacity-50'
                  : 'cursor-pointer hover:scale-110'
              }
              ${isSubmitting ? 'animate-pulse' : ''}
              ${star <= currentRating ? 'selected' : ''}
            `}
            disabled={disabled || isSubmitting}
            onClick={() => handleRatingClick(star)}
            onMouseEnter={() => handleMouseEnter(star)}
            aria-label={`评分 ${star} 星`}
            data-testid={`rating-star-${star}`}
          >
            <svg
              className={`
                w-full h-full transition-colors duration-200
                ${
                  star <= displayRating
                    ? 'text-yellow-400 fill-current'
                    : 'text-gray-300 hover:text-yellow-200'
                }
              `}
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"
              />
            </svg>
          </button>
        ))}

        {/* 评分文本 */}
        {showLabel && displayRating > 0 && (
          <span
            className={`
            ml-3 text-sm transition-opacity duration-200
            ${size === 'small' ? 'text-xs' : size === 'large' ? 'text-base' : 'text-sm'}
            ${isSubmitting ? 'opacity-50' : 'opacity-100'}
          `}
          >
            {RATING_LABELS[displayRating - 1]}
          </span>
        )}
      </div>

      {/* 状态信息 */}
      <div className="flex items-center space-x-2 text-xs">
        {isSubmitting && (
          <span className="text-blue-600 flex items-center space-x-1">
            <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24">
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
            <span>提交中...</span>
          </span>
        )}

        {hasSubmitted && !isSubmitting && (
          <span className="text-green-600 flex items-center space-x-1">
            <svg className="h-3 w-3" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                clipRule="evenodd"
              />
            </svg>
            <span>已评分</span>
          </span>
        )}

        {currentRating > 0 && !isSubmitting && (
          <span className="text-gray-500">当前: {currentRating}/5</span>
        )}
      </div>
    </div>
  )
}

export default RatingComponent
