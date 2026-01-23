/**
 * 综合反馈表单组件
 * 支持评分、点赞/踩、收藏、评论等多种反馈形式
 */

import React, { useState, useCallback, useRef } from 'react'
import { feedbackService } from '../../services/feedbackService'
import RatingComponent from './RatingComponent'
import LikeDislikeComponent from './LikeDislikeComponent'
import BookmarkComponent from './BookmarkComponent'

import { logger } from '../../utils/logger'
export interface FeedbackFormProps {
  itemId?: string
  userId: string
  sessionId: string
  title?: string
  showRating?: boolean
  showLikeDislike?: boolean
  showBookmark?: boolean
  showComment?: boolean
  commentPlaceholder?: string
  maxCommentLength?: number
  onSubmitSuccess?: (feedbackData: any) => void
  onSubmitError?: (error: Error) => void
  className?: string
  compact?: boolean
}

interface FeedbackState {
  rating: number
  likeState: 'none' | 'like' | 'dislike'
  bookmarked: boolean
  comment: string
}

export const FeedbackForm: React.FC<FeedbackFormProps> = ({
  itemId,
  userId,
  sessionId,
  title = '您的反馈',
  showRating = true,
  showLikeDislike = true,
  showBookmark = true,
  showComment = true,
  commentPlaceholder = '请分享您的想法...',
  maxCommentLength = 500,
  onSubmitSuccess,
  onSubmitError,
  className = '',
  compact = false,
}) => {
  const [feedbackState, setFeedbackState] = useState<FeedbackState>({
    rating: 0,
    likeState: 'none',
    bookmarked: false,
    comment: '',
  })
  const [isSubmittingComment, setIsSubmittingComment] = useState(false)
  const [commentSubmitted, setCommentSubmitted] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [validationErrors, setValidationErrors] = useState<string[]>([])

  const commentRef = useRef<HTMLTextAreaElement>(null)

  // 防作弊机制：检测快速重复操作
  const lastActionTime = useRef<number>(0)
  const actionCount = useRef<number>(0)
  const RATE_LIMIT_WINDOW = 60000 // 1分钟
  const MAX_ACTIONS_PER_WINDOW = 10

  // 反作弊检查
  const checkAntiCheat = useCallback((): boolean => {
    const now = Date.now()

    // 重置计数器如果超过时间窗口
    if (now - lastActionTime.current > RATE_LIMIT_WINDOW) {
      actionCount.current = 0
    }

    actionCount.current++
    lastActionTime.current = now

    if (actionCount.current > MAX_ACTIONS_PER_WINDOW) {
      setValidationErrors(prev => [...prev, '操作过于频繁，请稍后再试'])
      return false
    }

    return true
  }, [])

  // 验证评论内容
  const validateComment = useCallback(
    (comment: string): boolean => {
      const errors: string[] = []

      // 长度检查
      if (comment.length > maxCommentLength) {
        errors.push(`评论长度不能超过${maxCommentLength}字符`)
      }

      // 内容质量检查
      if (comment.trim().length < 3) {
        errors.push('评论内容至少需要3个字符')
      }

      // 检测垃圾内容（简单规则）
      const spamPatterns = [
        /(.)\1{5,}/, // 重复字符
        /^[!@#$%^&*()]+$/, // 仅符号
        /^[0-9]+$/, // 仅数字
      ]

      const isSpam = spamPatterns.some(pattern => pattern.test(comment))
      if (isSpam) {
        errors.push('评论内容不符合要求')
      }

      setValidationErrors(errors)
      return errors.length === 0
    },
    [maxCommentLength]
  )

  // 提交评论
  const handleCommentSubmit = useCallback(async () => {
    if (!checkAntiCheat()) return
    if (!validateComment(feedbackState.comment)) return

    try {
      setIsSubmittingComment(true)

      await feedbackService.submitComment(
        userId,
        itemId || 'unknown',
        feedbackState.comment,
        sessionId,
        {
          component: 'FeedbackForm',
          characterCount: feedbackState.comment.length,
          timestamp: Date.now(),
          hasRating: feedbackState.rating > 0,
          hasLikeState: feedbackState.likeState !== 'none',
          hasBookmark: feedbackState.bookmarked,
        }
      )

      setCommentSubmitted(true)
      setValidationErrors([])

      onSubmitSuccess?.({
        type: 'comment',
        comment: feedbackState.comment,
        ...feedbackState,
      })
    } catch (error) {
      logger.error('提交评论失败:', error)
      onSubmitError?.(error as Error)
    } finally {
      setIsSubmittingComment(false)
    }
  }, [
    checkAntiCheat,
    validateComment,
    feedbackState,
    userId,
    itemId,
    sessionId,
    onSubmitSuccess,
    onSubmitError,
  ])

  // 处理评分变化
  const handleRatingChange = useCallback((rating: number) => {
    setFeedbackState(prev => ({ ...prev, rating }))
  }, [])

  // 处理点赞/踩状态变化
  const handleLikeStateChange = useCallback(
    (likeState: 'none' | 'like' | 'dislike') => {
      setFeedbackState(prev => ({ ...prev, likeState }))
    },
    []
  )

  // 处理收藏状态变化
  const handleBookmarkChange = useCallback((bookmarked: boolean) => {
    setFeedbackState(prev => ({ ...prev, bookmarked }))
  }, [])

  // 处理评论内容变化
  const handleCommentChange = useCallback(
    (event: React.ChangeEvent<HTMLTextAreaElement>) => {
      const comment = event.target.value
      setFeedbackState(prev => ({ ...prev, comment }))

      // 清除验证错误
      if (validationErrors.length > 0) {
        setValidationErrors([])
      }
    },
    [validationErrors.length]
  )

  // 获取反馈摘要
  const getFeedbackSummary = () => {
    const summary = []
    if (feedbackState.rating > 0)
      summary.push(`评分: ${feedbackState.rating}/5`)
    if (feedbackState.likeState !== 'none')
      summary.push(
        `态度: ${feedbackState.likeState === 'like' ? '喜欢' : '不喜欢'}`
      )
    if (feedbackState.bookmarked) summary.push('已收藏')
    if (feedbackState.comment.trim()) summary.push('有评论')
    return summary
  }

  const containerClass = compact
    ? 'space-y-3 p-3 bg-gray-50 rounded-lg'
    : 'space-y-6 p-6 bg-white border border-gray-200 rounded-xl shadow-sm'

  return (
    <div className={`${containerClass} ${className}`}>
      {/* 标题 */}
      {!compact && (
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-medium text-gray-900">{title}</h3>
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-sm text-blue-600 hover:text-blue-800 transition-colors"
          >
            {showAdvanced ? '收起' : '更多选项'}
          </button>
        </div>
      )}

      {/* 快速反馈区域 */}
      <div className="flex flex-wrap items-center gap-4">
        {showRating && (
          <div className="flex flex-col space-y-1">
            {!compact && (
              <label className="text-sm font-medium text-gray-700">评分</label>
            )}
            <RatingComponent
              itemId={itemId}
              userId={userId}
              sessionId={sessionId}
              size={compact ? 'small' : 'medium'}
              showLabel={!compact}
              onRatingChange={handleRatingChange}
              onSubmitSuccess={rating =>
                onSubmitSuccess?.({ type: 'rating', rating, ...feedbackState })
              }
              onSubmitError={onSubmitError}
            />
          </div>
        )}

        {showLikeDislike && (
          <div className="flex flex-col space-y-1">
            {!compact && (
              <label className="text-sm font-medium text-gray-700">喜好</label>
            )}
            <LikeDislikeComponent
              itemId={itemId}
              userId={userId}
              sessionId={sessionId}
              size={compact ? 'small' : 'medium'}
              onStateChange={handleLikeStateChange}
              onSubmitSuccess={action =>
                onSubmitSuccess?.({ type: 'like', action, ...feedbackState })
              }
              onSubmitError={onSubmitError}
            />
          </div>
        )}

        {showBookmark && (
          <div className="flex flex-col space-y-1">
            {!compact && (
              <label className="text-sm font-medium text-gray-700">收藏</label>
            )}
            <BookmarkComponent
              itemId={itemId}
              userId={userId}
              sessionId={sessionId}
              size={compact ? 'small' : 'medium'}
              showLabel={!compact}
              onBookmarkChange={handleBookmarkChange}
              onSubmitSuccess={bookmarked =>
                onSubmitSuccess?.({
                  type: 'bookmark',
                  bookmarked,
                  ...feedbackState,
                })
              }
              onSubmitError={onSubmitError}
            />
          </div>
        )}
      </div>

      {/* 评论区域 */}
      {showComment && (showAdvanced || compact) && (
        <div className="space-y-3">
          <label className="block text-sm font-medium text-gray-700">
            评论 {!compact && '(可选)'}
          </label>

          <div className="space-y-2">
            <textarea
              ref={commentRef}
              value={feedbackState.comment}
              onChange={handleCommentChange}
              placeholder={commentPlaceholder}
              maxLength={maxCommentLength}
              disabled={commentSubmitted || isSubmittingComment}
              className={`
                w-full px-3 py-2 border border-gray-300 rounded-md resize-none
                focus:ring-2 focus:ring-blue-500 focus:border-transparent
                disabled:bg-gray-100 disabled:cursor-not-allowed
                transition-colors duration-200
                ${compact ? 'h-20 text-sm' : 'h-24'}
                ${validationErrors.length > 0 ? 'border-red-300 ring-1 ring-red-300' : ''}
              `}
              aria-describedby="comment-help"
            />

            <div className="flex items-center justify-between text-xs text-gray-500">
              <span id="comment-help">
                {feedbackState.comment.length}/{maxCommentLength} 字符
              </span>

              {!commentSubmitted && feedbackState.comment.trim() && (
                <button
                  onClick={handleCommentSubmit}
                  disabled={isSubmittingComment || validationErrors.length > 0}
                  className={`
                    px-3 py-1 rounded-md text-xs font-medium
                    transition-colors duration-200
                    ${
                      isSubmittingComment || validationErrors.length > 0
                        ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                        : 'bg-blue-600 text-white hover:bg-blue-700'
                    }
                  `}
                >
                  {isSubmittingComment ? '提交中...' : '提交评论'}
                </button>
              )}
            </div>
          </div>

          {/* 验证错误 */}
          {validationErrors.length > 0 && (
            <div className="text-xs text-red-600 space-y-1">
              {validationErrors.map((error, index) => (
                <div key={index}>• {error}</div>
              ))}
            </div>
          )}

          {/* 评论成功提示 */}
          {commentSubmitted && (
            <div className="flex items-center space-x-2 text-xs text-green-600">
              <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                  clipRule="evenodd"
                />
              </svg>
              <span>评论已提交</span>
            </div>
          )}
        </div>
      )}

      {/* 反馈摘要 */}
      {!compact && showAdvanced && (
        <div className="pt-4 border-t border-gray-200">
          <div className="text-sm text-gray-600">
            <span className="font-medium">反馈摘要: </span>
            {getFeedbackSummary().length > 0
              ? getFeedbackSummary().join(', ')
              : '暂无反馈'}
          </div>
        </div>
      )}
    </div>
  )
}

export default FeedbackForm
