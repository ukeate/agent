import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { Alert, Card, Input, Space, Typography } from 'antd'
import FeedbackForm from '../components/feedback/FeedbackForm'

const { Title, Text } = Typography

const STORAGE_KEYS = {
  userId: 'ai-agent-feedback-user',
  itemId: 'ai-agent-feedback-item',
}

const buildSessionId = () =>
  `session_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`

const FeedbackSystemPage: React.FC = () => {
  const [userId, setUserId] = useState(() => {
    if (typeof window === 'undefined') return 'demo-user'
    return localStorage.getItem(STORAGE_KEYS.userId) || 'demo-user'
  })
  const [itemId, setItemId] = useState(() => {
    if (typeof window === 'undefined') return 'demo-item'
    return localStorage.getItem(STORAGE_KEYS.itemId) || 'demo-item'
  })
  const [sessionId] = useState(buildSessionId)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [isMobile, setIsMobile] = useState(false)

  useEffect(() => {
    if (typeof window === 'undefined') return
    localStorage.setItem(STORAGE_KEYS.userId, userId)
  }, [userId])

  useEffect(() => {
    if (typeof window === 'undefined') return
    localStorage.setItem(STORAGE_KEYS.itemId, itemId)
  }, [itemId])

  useEffect(() => {
    if (typeof window === 'undefined') return
    const update = () => setIsMobile(window.innerWidth <= 480)
    update()
    window.addEventListener('resize', update)
    return () => window.removeEventListener('resize', update)
  }, [])

  const handleSubmitSuccess = useCallback((payload: any) => {
    const type = payload?.type
    if (type === 'rating') {
      setSuccessMessage(`评分已提交：${payload?.rating || 0}/5`)
    } else if (type === 'like') {
      setSuccessMessage(
        payload?.action === 'dislike' ? '已提交点踩' : '已提交点赞'
      )
    } else if (type === 'bookmark') {
      setSuccessMessage(payload?.bookmarked ? '已添加收藏' : '已取消收藏')
    } else if (type === 'comment') {
      setSuccessMessage('评论已提交')
    } else {
      setSuccessMessage('反馈已提交')
    }
    setErrorMessage(null)
  }, [])

  const handleSubmitError = useCallback((error: Error) => {
    setErrorMessage(error?.message || '提交失败，请稍后重试')
    setSuccessMessage(null)
  }, [])

  const formKey = useMemo(
    () => `${userId}-${itemId}-${sessionId}`,
    [userId, itemId, sessionId]
  )

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <div>
          <Title level={3} style={{ marginBottom: 4 }}>
            用户反馈系统
          </Title>
          <Text type="secondary">
            提交评分、点赞、收藏与评论，实时沉淀用户偏好信号。
          </Text>
        </div>

        <Card>
          <Space
            direction="vertical"
            size="middle"
            style={{ width: '100%' }}
          >
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
              <div style={{ minWidth: 220, flex: 1 }}>
                <Text type="secondary">用户ID</Text>
                <Input
                  value={userId}
                  onChange={event => setUserId(event.target.value)}
                  placeholder="输入用户ID"
                  data-testid="feedback-user-id"
                />
              </div>
              <div style={{ minWidth: 220, flex: 1 }}>
                <Text type="secondary">物品ID</Text>
                <Input
                  value={itemId}
                  onChange={event => setItemId(event.target.value)}
                  placeholder="输入物品ID"
                  data-testid="feedback-item-id"
                />
              </div>
            </div>
            <Text type="secondary" style={{ fontSize: 12 }}>
              会话ID：{sessionId}
            </Text>
          </Space>
        </Card>

        {errorMessage && (
          <Alert
            type="error"
            message={errorMessage}
            showIcon
            data-testid="feedback-error"
          />
        )}
        {successMessage && (
          <Alert
            type="success"
            message={successMessage}
            showIcon
            data-testid="feedback-success"
          />
        )}

        <FeedbackForm
          key={formKey}
          userId={userId || 'demo-user'}
          itemId={itemId || 'demo-item'}
          sessionId={sessionId}
          onSubmitSuccess={handleSubmitSuccess}
          onSubmitError={handleSubmitError}
          className={isMobile ? 'mobile' : ''}
          touchMode={isMobile}
        />
      </Space>
    </div>
  )
}

export default FeedbackSystemPage
