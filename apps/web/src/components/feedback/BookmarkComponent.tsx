/**
 * 收藏反馈组件
 * 支持收藏/取消收藏操作
 */

import React, { useState, useCallback } from 'react'
import { feedbackService } from '../../services/feedbackService'

import { logger } from '../../utils/logger'
export interface BookmarkComponentProps {
  itemId?: string
  userId: string
  sessionId: string
  initialBookmarked?: boolean
  size?: 'small' | 'medium' | 'large'
  disabled?: boolean
  showLabel?: boolean
  onBookmarkChange?: (bookmarked: boolean) => void
  onSubmitSuccess?: (bookmarked: boolean) => void
  onSubmitError?: (error: Error) => void
}

export const BookmarkComponent: React.FC<BookmarkComponentProps> = ({
  itemId,
  userId,
  sessionId,
  initialBookmarked = false,
  size = 'medium',
  disabled = false,
  showLabel = true,
  onBookmarkChange,
  onSubmitSuccess,
  onSubmitError,
}) => {
  const [isBookmarked, setIsBookmarked] = useState(initialBookmarked)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isHovered, setIsHovered] = useState(false)

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
        return 'p-1.5'
      case 'large':
        return 'p-3'
      default:
        return 'p-2'
    }
  }

  // 处理收藏操作
  const handleBookmark = useCallback(async () => {
    if (disabled || isSubmitting) return

    const previousState = isBookmarked
    const newState = !isBookmarked

    // 乐观更新UI
    setIsBookmarked(newState)
    onBookmarkChange?.(newState)

    try {
      setIsSubmitting(true)

      await feedbackService.submitBookmark(
        userId,
        itemId || 'unknown',
        newState,
        sessionId,
        {
          component: 'BookmarkComponent',
          action: newState ? 'bookmark' : 'unbookmark',
          previousState,
          timestamp: Date.now(),
        }
      )

      onSubmitSuccess?.(newState)
    } catch (error) {
      logger.error('提交收藏反馈失败:', error)

      // 回滚状态
      setIsBookmarked(previousState)
      onBookmarkChange?.(previousState)

      onSubmitError?.(error as Error)
    } finally {
      setIsSubmitting(false)
    }
  }, [
    disabled,
    isSubmitting,
    isBookmarked,
    userId,
    itemId,
    sessionId,
    onBookmarkChange,
    onSubmitSuccess,
    onSubmitError,
  ])

  // 获取按钮样式
  const getButtonStyle = () => {
    const baseStyle = `
      ${getButtonSize()}
      rounded-full transition-all duration-200 ease-in-out
      flex items-center justify-center space-x-2
      ${disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer hover:scale-105'}
      ${isSubmitting ? 'animate-pulse' : ''}
    `

    if (isBookmarked) {
      return `${baseStyle} bg-yellow-100 text-yellow-600 ring-2 ring-yellow-300 hover:bg-yellow-200`
    }

    if (isHovered) {
      return `${baseStyle} bg-yellow-50 text-yellow-500 hover:bg-yellow-100`
    }

    return `${baseStyle} bg-gray-100 text-gray-500 hover:bg-yellow-50 hover:text-yellow-500`
  }

  // 获取图标填充样式
  const getIconFill = () => {
    if (isBookmarked) {
      return 'currentColor'
    }
    return isHovered ? 'rgba(234, 179, 8, 0.3)' : 'none'
  }

  // 获取文本样式
  const getTextSize = () => {
    switch (size) {
      case 'small':
        return 'text-xs'
      case 'large':
        return 'text-base'
      default:
        return 'text-sm'
    }
  }

  return (
    <div className="flex items-center">
      <button
        type="button"
        className={getButtonStyle()}
        disabled={disabled || isSubmitting}
        onClick={handleBookmark}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        aria-label={isBookmarked ? '取消收藏' : '收藏'}
        title={isBookmarked ? '取消收藏' : '收藏'}
      >
        {/* 收藏图标 */}
        <svg
          className={`${getIconSize()} transition-all duration-200 ${
            isBookmarked || isHovered ? 'scale-110' : ''
          }`}
          fill={getIconFill()}
          stroke="currentColor"
          strokeWidth={1.5}
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M17.593 3.322c1.1.128 1.907 1.077 1.907 2.185V21L12 17.25 4.5 21V5.507c0-1.108.806-2.057 1.907-2.185a48.507 48.507 0 0111.186 0z"
          />
        </svg>

        {/* 文本标签 */}
        {showLabel && size !== 'small' && (
          <span
            className={`${getTextSize()} font-medium transition-opacity duration-200`}
          >
            {isBookmarked ? '已收藏' : '收藏'}
          </span>
        )}

        {/* 加载指示器 */}
        {isSubmitting && (
          <svg className="animate-spin h-3 w-3 ml-1" viewBox="0 0 24 24">
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
        )}
      </button>

      {/* 状态提示 */}
      {isBookmarked && !isSubmitting && size === 'large' && (
        <div className="ml-2 flex items-center space-x-1 text-xs text-yellow-600">
          <svg className="h-3 w-3" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
              clipRule="evenodd"
            />
          </svg>
          <span>已保存到收藏</span>
        </div>
      )}
    </div>
  )
}

export default BookmarkComponent
