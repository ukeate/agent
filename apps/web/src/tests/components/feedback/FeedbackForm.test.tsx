/**
 * FeedbackForm 组件测试套件
 *
 * 测试用户反馈表单的所有功能，包括评分、点赞、收藏、评论等交互。
 * 覆盖验证、防作弊、状态管理等核心功能。
 */

import React from 'react'
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest'

import { FeedbackForm } from '../../../components/feedback/FeedbackForm'
import { feedbackService } from '../../../services/feedbackService'

// Mock 子组件
vi.mock('../../../components/feedback/RatingComponent', () => ({
  default: ({ onRatingChange, onSubmitSuccess }: any) => (
    <div data-testid="rating-component">
      <button
        onClick={() => {
          onRatingChange(5)
          onSubmitSuccess?.(5)
        }}
        data-testid="rating-5"
      >
        5 Stars
      </button>
    </div>
  ),
}))

vi.mock('../../../components/feedback/LikeDislikeComponent', () => ({
  default: ({ onStateChange, onSubmitSuccess }: any) => (
    <div data-testid="like-dislike-component">
      <button
        onClick={() => {
          onStateChange('like')
          onSubmitSuccess?.('like')
        }}
        data-testid="like-button"
      >
        Like
      </button>
      <button
        onClick={() => {
          onStateChange('dislike')
          onSubmitSuccess?.('dislike')
        }}
        data-testid="dislike-button"
      >
        Dislike
      </button>
    </div>
  ),
}))

vi.mock('../../../components/feedback/BookmarkComponent', () => ({
  default: ({ onBookmarkChange, onSubmitSuccess }: any) => (
    <div data-testid="bookmark-component">
      <button
        onClick={() => {
          onBookmarkChange(true)
          onSubmitSuccess?.(true)
        }}
        data-testid="bookmark-button"
      >
        Bookmark
      </button>
    </div>
  ),
}))

// Mock feedbackService
vi.mock('../../../services/feedbackService', () => ({
  feedbackService: {
    submitComment: vi.fn(),
    submitRating: vi.fn(),
    submitLike: vi.fn(),
    submitDislike: vi.fn(),
    submitBookmark: vi.fn(),
  },
}))

describe('FeedbackForm', () => {
  const defaultProps = {
    itemId: 'test-item-123',
    userId: 'user-456',
    sessionId: 'session-789',
  }

  const mockFeedbackService = vi.mocked(feedbackService)

  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  describe('基本渲染', () => {
    it('应该渲染默认的反馈表单', () => {
      render(<FeedbackForm {...defaultProps} />)

      expect(screen.getByText('您的反馈')).toBeInTheDocument()
      expect(screen.getByTestId('rating-component')).toBeInTheDocument()
      expect(screen.getByTestId('like-dislike-component')).toBeInTheDocument()
      expect(screen.getByTestId('bookmark-component')).toBeInTheDocument()
      expect(screen.getByText('更多选项')).toBeInTheDocument()
    })

    it('应该支持紧凑模式渲染', () => {
      render(<FeedbackForm {...defaultProps} compact />)

      // 紧凑模式不显示标题和更多选项按钮
      expect(screen.queryByText('您的反馈')).not.toBeInTheDocument()
      expect(screen.queryByText('更多选项')).not.toBeInTheDocument()

      // 但应该显示评论区域
      expect(screen.getByRole('textbox')).toBeInTheDocument()
    })

    it('应该支持自定义标题', () => {
      const customTitle = '请评价此产品'
      render(<FeedbackForm {...defaultProps} title={customTitle} />)

      expect(screen.getByText(customTitle)).toBeInTheDocument()
    })

    it('应该支持隐藏特定组件', () => {
      render(
        <FeedbackForm
          {...defaultProps}
          showRating={false}
          showLikeDislike={false}
          showBookmark={false}
          showComment={false}
        />
      )

      expect(screen.queryByTestId('rating-component')).not.toBeInTheDocument()
      expect(
        screen.queryByTestId('like-dislike-component')
      ).not.toBeInTheDocument()
      expect(screen.queryByTestId('bookmark-component')).not.toBeInTheDocument()
      expect(screen.queryByRole('textbox')).not.toBeInTheDocument()
    })
  })

  describe('评论功能', () => {
    it('应该正确处理评论输入', async () => {
      const user = userEvent.setup()
      render(<FeedbackForm {...defaultProps} />)

      // 展开高级选项显示评论区域
      await user.click(screen.getByText('更多选项'))

      const textarea = screen.getByRole('textbox')
      await user.type(textarea, '这是一个测试评论')

      expect(textarea).toHaveValue('这是一个测试评论')
      expect(screen.getByText('9/500 字符')).toBeInTheDocument()
    })

    it('应该正确提交评论', async () => {
      const user = userEvent.setup()
      const onSubmitSuccess = vi.fn()
      mockFeedbackService.submitComment.mockResolvedValueOnce({})

      render(
        <FeedbackForm {...defaultProps} onSubmitSuccess={onSubmitSuccess} />
      )

      // 展开高级选项
      await user.click(screen.getByText('更多选项'))

      const textarea = screen.getByRole('textbox')
      await user.type(textarea, '这是一个很好的产品')

      // 点击提交按钮
      await user.click(screen.getByText('提交评论'))

      await waitFor(() => {
        expect(mockFeedbackService.submitComment).toHaveBeenCalledWith(
          'user-456',
          'test-item-123',
          '这是一个很好的产品',
          'session-789',
          expect.objectContaining({
            component: 'FeedbackForm',
            characterCount: 8,
            timestamp: expect.any(Number),
          })
        )
      })

      expect(onSubmitSuccess).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'comment',
          comment: '这是一个很好的产品',
        })
      )

      expect(screen.getByText('评论已提交')).toBeInTheDocument()
    })

    it('应该验证评论长度', async () => {
      const user = userEvent.setup()
      render(<FeedbackForm {...defaultProps} maxCommentLength={10} />)

      await user.click(screen.getByText('更多选项'))

      const textarea = screen.getByRole('textbox')
      await user.type(textarea, '这是一个超过长度限制的评论内容')

      // 尝试提交
      await user.click(screen.getByText('提交评论'))

      expect(screen.getByText('• 评论长度不能超过10字符')).toBeInTheDocument()
      expect(mockFeedbackService.submitComment).not.toHaveBeenCalled()
    })

    it('应该验证评论最小长度', async () => {
      const user = userEvent.setup()
      render(<FeedbackForm {...defaultProps} />)

      await user.click(screen.getByText('更多选项'))

      const textarea = screen.getByRole('textbox')
      await user.type(textarea, '短') // 只有1个字符

      await user.click(screen.getByText('提交评论'))

      expect(screen.getByText('• 评论内容至少需要3个字符')).toBeInTheDocument()
      expect(mockFeedbackService.submitComment).not.toHaveBeenCalled()
    })

    it('应该检测垃圾内容', async () => {
      const user = userEvent.setup()
      render(<FeedbackForm {...defaultProps} />)

      await user.click(screen.getByText('更多选项'))

      const textarea = screen.getByRole('textbox')

      // 测试重复字符
      await user.clear(textarea)
      await user.type(textarea, 'aaaaaaaaa') // 9个重复字符
      await user.click(screen.getByText('提交评论'))

      expect(screen.getByText('• 评论内容不符合要求')).toBeInTheDocument()

      // 清除输入并测试仅符号
      await user.clear(textarea)
      await user.type(textarea, '!@#$%')
      await user.click(screen.getByText('提交评论'))

      expect(screen.getByText('• 评论内容不符合要求')).toBeInTheDocument()
    })

    it('应该处理评论提交错误', async () => {
      const user = userEvent.setup()
      const onSubmitError = vi.fn()
      const error = new Error('网络错误')

      mockFeedbackService.submitComment.mockRejectedValueOnce(error)

      render(<FeedbackForm {...defaultProps} onSubmitError={onSubmitError} />)

      await user.click(screen.getByText('更多选项'))

      const textarea = screen.getByRole('textbox')
      await user.type(textarea, '测试评论内容')
      await user.click(screen.getByText('提交评论'))

      await waitFor(() => {
        expect(onSubmitError).toHaveBeenCalledWith(error)
      })

      // 不应该显示成功提示
      expect(screen.queryByText('评论已提交')).not.toBeInTheDocument()
    })
  })

  describe('防作弊机制', () => {
    it('应该限制快速重复操作', async () => {
      const user = userEvent.setup()
      render(<FeedbackForm {...defaultProps} />)

      await user.click(screen.getByText('更多选项'))

      const textarea = screen.getByRole('textbox')

      // 快速提交多次（超过限制）
      for (let i = 0; i < 12; i++) {
        await user.clear(textarea)
        await user.type(textarea, `测试评论 ${i}`)

        const submitButton = screen.queryByText('提交评论')
        if (submitButton) {
          await user.click(submitButton)
        }
      }

      // 应该显示频率限制错误
      await waitFor(() => {
        expect(
          screen.getByText('• 操作过于频繁，请稍后再试')
        ).toBeInTheDocument()
      })
    })

    it('应该在时间窗口重置后恢复正常', async () => {
      const user = userEvent.setup()
      render(<FeedbackForm {...defaultProps} />)

      await user.click(screen.getByText('更多选项'))

      const textarea = screen.getByRole('textbox')

      // 快速操作触发限制
      for (let i = 0; i < 11; i++) {
        await user.clear(textarea)
        await user.type(textarea, `测试 ${i}`)
        const submitButton = screen.queryByText('提交评论')
        if (submitButton) {
          await user.click(submitButton)
        }
      }

      // 确认限制被触发
      expect(screen.getByText('• 操作过于频繁，请稍后再试')).toBeInTheDocument()

      // 前进时间超过限制窗口（1分钟）
      act(() => {
        vi.advanceTimersByTime(61000)
      })

      // 清除之前的错误状态
      await user.clear(textarea)
      await user.type(textarea, '重置后的评论')

      // 现在应该可以正常提交
      mockFeedbackService.submitComment.mockResolvedValueOnce({})
      await user.click(screen.getByText('提交评论'))

      await waitFor(() => {
        expect(mockFeedbackService.submitComment).toHaveBeenCalledWith(
          expect.any(String),
          expect.any(String),
          '重置后的评论',
          expect.any(String),
          expect.any(Object)
        )
      })
    })
  })

  describe('状态管理', () => {
    it('应该正确管理反馈状态', async () => {
      const user = userEvent.setup()
      const onSubmitSuccess = vi.fn()

      render(
        <FeedbackForm {...defaultProps} onSubmitSuccess={onSubmitSuccess} />
      )

      // 设置评分
      await user.click(screen.getByTestId('rating-5'))
      expect(onSubmitSuccess).toHaveBeenLastCalledWith(
        expect.objectContaining({
          type: 'rating',
          rating: 5,
        })
      )

      // 设置喜好
      await user.click(screen.getByTestId('like-button'))
      expect(onSubmitSuccess).toHaveBeenLastCalledWith(
        expect.objectContaining({
          type: 'like',
          action: 'like',
        })
      )

      // 设置收藏
      await user.click(screen.getByTestId('bookmark-button'))
      expect(onSubmitSuccess).toHaveBeenLastCalledWith(
        expect.objectContaining({
          type: 'bookmark',
          bookmarked: true,
        })
      )
    })

    it('应该显示反馈摘要', async () => {
      const user = userEvent.setup()
      render(<FeedbackForm {...defaultProps} />)

      // 展开高级选项
      await user.click(screen.getByText('更多选项'))

      // 初始状态应该显示无反馈
      expect(screen.getByText('反馈摘要: 暂无反馈')).toBeInTheDocument()

      // 设置一些反馈
      await user.click(screen.getByTestId('rating-5'))
      await user.click(screen.getByTestId('like-button'))
      await user.click(screen.getByTestId('bookmark-button'))

      // 应该更新摘要
      await waitFor(() => {
        expect(
          screen.getByText(/反馈摘要: .*评分: 5\/5.*喜欢.*已收藏/)
        ).toBeInTheDocument()
      })
    })
  })

  describe('无障碍性', () => {
    it('应该有正确的 ARIA 标签', () => {
      render(<FeedbackForm {...defaultProps} compact />)

      const textarea = screen.getByRole('textbox')
      expect(textarea).toHaveAttribute('aria-describedby', 'comment-help')

      const helpText = screen.getByText('0/500 字符')
      expect(helpText).toHaveAttribute('id', 'comment-help')
    })

    it('应该正确处理禁用状态', async () => {
      const user = userEvent.setup()
      mockFeedbackService.submitComment.mockResolvedValueOnce({})

      render(<FeedbackForm {...defaultProps} compact />)

      const textarea = screen.getByRole('textbox')
      await user.type(textarea, '测试评论')

      // 提交后 textarea 应该被禁用
      await user.click(screen.getByText('提交评论'))

      await waitFor(() => {
        expect(textarea).toBeDisabled()
      })
    })
  })

  describe('边界情况', () => {
    it('应该处理缺少 itemId 的情况', async () => {
      const user = userEvent.setup()
      mockFeedbackService.submitComment.mockResolvedValueOnce({})

      render(
        <FeedbackForm
          userId="user-456"
          sessionId="session-789"
          // 没有 itemId
        />
      )

      await user.click(screen.getByText('更多选项'))

      const textarea = screen.getByRole('textbox')
      await user.type(textarea, '测试评论')
      await user.click(screen.getByText('提交评论'))

      await waitFor(() => {
        expect(mockFeedbackService.submitComment).toHaveBeenCalledWith(
          'user-456',
          'unknown', // 应该使用默认值
          '测试评论',
          'session-789',
          expect.any(Object)
        )
      })
    })

    it('应该处理空的回调函数', async () => {
      const user = userEvent.setup()
      mockFeedbackService.submitComment.mockResolvedValueOnce({})

      render(<FeedbackForm {...defaultProps} />)

      await user.click(screen.getByText('更多选项'))

      const textarea = screen.getByRole('textbox')
      await user.type(textarea, '测试评论')

      // 即使没有回调函数也不应该报错
      await user.click(screen.getByText('提交评论'))

      await waitFor(() => {
        expect(mockFeedbackService.submitComment).toHaveBeenCalled()
      })
    })

    it('应该正确清除验证错误', async () => {
      const user = userEvent.setup()
      render(<FeedbackForm {...defaultProps} maxCommentLength={5} />)

      await user.click(screen.getByText('更多选项'))

      const textarea = screen.getByRole('textbox')

      // 输入过长内容触发错误
      await user.type(textarea, '这是一个很长的评论')
      await user.click(screen.getByText('提交评论'))

      expect(screen.getByText('• 评论长度不能超过5字符')).toBeInTheDocument()

      // 修改为有效内容，错误应该被清除
      await user.clear(textarea)
      await user.type(textarea, '短评')

      expect(
        screen.queryByText('• 评论长度不能超过5字符')
      ).not.toBeInTheDocument()
    })
  })

  describe('性能测试', () => {
    it('应该避免不必要的重新渲染', async () => {
      const user = userEvent.setup()
      const onSubmitSuccess = vi.fn()

      const { rerender } = render(
        <FeedbackForm {...defaultProps} onSubmitSuccess={onSubmitSuccess} />
      )

      // 多次重新渲染相同的 props
      for (let i = 0; i < 5; i++) {
        rerender(
          <FeedbackForm {...defaultProps} onSubmitSuccess={onSubmitSuccess} />
        )
      }

      // 组件应该仍然可用
      expect(screen.getByText('您的反馈')).toBeInTheDocument()

      // 交互应该正常工作
      await user.click(screen.getByTestId('rating-5'))
      expect(onSubmitSuccess).toHaveBeenCalled()
    })
  })
})
