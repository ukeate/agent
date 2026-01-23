import { useRef, useEffect, useCallback, useState } from 'react'

interface UseSmartAutoScrollOptions {
  /**
   * 消息数组，当数组长度变化时触发滚动检查
   */
  messages: any[]

  /**
   * 是否启用自动滚动
   */
  enabled?: boolean

  /**
   * 距离底部多少像素内被认为是"在底部"
   */
  threshold?: number

  /**
   * 滚动行为
   */
  behavior?: ScrollBehavior

  /**
   * 会话切换或重新加载时重置初始滚动
   */
  resetKey?: string | number | null
}

export const useSmartAutoScroll = ({
  messages,
  enabled = true,
  threshold = 100,
  behavior = 'smooth',
  resetKey,
}: UseSmartAutoScrollOptions) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const [isAtBottom, setIsAtBottom] = useState(true)
  const isUserScrollingRef = useRef(false)
  const isAutoScrollingRef = useRef(false)
  const lastScrollTopRef = useRef(0)
  const scrollTimeoutRef = useRef<ReturnType<typeof setTimeout>>()
  const autoScrollTimeoutRef = useRef<ReturnType<typeof setTimeout>>()
  const initialScrollRef = useRef(true)
  const lastResetKeyRef = useRef(resetKey)
  const lastMessage = messages[messages.length - 1]
  const lastMessageSignature = lastMessage
    ? `${lastMessage.id ?? ''}:${lastMessage.content?.length ?? 0}:${lastMessage.toolCalls?.length ?? 0}:${lastMessage.reasoningSteps?.length ?? 0}`
    : ''

  // 检查是否在底部附近
  const isNearBottom = useCallback(() => {
    if (!containerRef.current) return false

    const { scrollTop, scrollHeight, clientHeight } = containerRef.current
    return scrollHeight - scrollTop - clientHeight <= threshold
  }, [threshold])

  // 滚动到底部
  const resetAutoScrolling = useCallback(() => {
    if (autoScrollTimeoutRef.current) {
      clearTimeout(autoScrollTimeoutRef.current)
    }
    const delay = behavior === 'smooth' ? 300 : 0
    autoScrollTimeoutRef.current = setTimeout(() => {
      isAutoScrollingRef.current = false
    }, delay)
  }, [behavior])

  const scrollToBottom = useCallback(() => {
    if (!containerRef.current) return

    isAutoScrollingRef.current = true
    containerRef.current.scrollTo({
      top: containerRef.current.scrollHeight,
      behavior,
    })
    setIsAtBottom(true)
    resetAutoScrolling()
  }, [behavior, resetAutoScrolling])

  // 处理用户手动滚动
  const handleScroll = useCallback(() => {
    if (!containerRef.current) return

    const currentScrollTop = containerRef.current.scrollTop
    const nearBottom = isNearBottom()
    setIsAtBottom(nearBottom)
    if (isAutoScrollingRef.current && nearBottom) {
      lastScrollTopRef.current = currentScrollTop
      return
    }

    // 检测用户是否主动滚动（而不是自动滚动）
    if (Math.abs(currentScrollTop - lastScrollTopRef.current) > 5) {
      isUserScrollingRef.current = true

      // 清除之前的定时器
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current)
      }

      // 1秒后重置用户滚动状态，允许自动滚动
      scrollTimeoutRef.current = setTimeout(() => {
        isUserScrollingRef.current = false
      }, 1000)
    }

    lastScrollTopRef.current = currentScrollTop
  }, [isNearBottom])

  useEffect(() => {
    if (lastResetKeyRef.current === resetKey) return
    lastResetKeyRef.current = resetKey
    initialScrollRef.current = true
  }, [resetKey])

  // 监听消息变化，智能决定是否自动滚动
  useEffect(() => {
    if (!enabled || !containerRef.current || messages.length === 0) {
      return
    }

    // 延迟执行，确保DOM已更新
    const scrollTimeout = setTimeout(() => {
      // 只有在用户当前在底部附近且没有主动滚动时才自动滚动
      if (!isUserScrollingRef.current && isNearBottom()) {
        scrollToBottom()
        return
      }
      setIsAtBottom(isNearBottom())
    }, 50)

    return () => clearTimeout(scrollTimeout)
  }, [
    messages.length,
    lastMessageSignature,
    enabled,
    isNearBottom,
    scrollToBottom,
  ])

  // 绑定滚动事件监听
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    container.addEventListener('scroll', handleScroll, { passive: true })

    return () => {
      container.removeEventListener('scroll', handleScroll)
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current)
      }
      if (autoScrollTimeoutRef.current) {
        clearTimeout(autoScrollTimeoutRef.current)
      }
    }
  }, [handleScroll])

  // 首次进入或会话切换时滚动到底部
  useEffect(() => {
    if (!enabled || messages.length === 0 || !initialScrollRef.current) return
    initialScrollRef.current = false
    const initialScrollTimeout = setTimeout(() => {
      scrollToBottom()
    }, 100)
    return () => clearTimeout(initialScrollTimeout)
  }, [enabled, messages.length, resetKey, scrollToBottom])

  return {
    containerRef,
    scrollToBottom,
    isNearBottom,
    isAtBottom,
    isUserScrolling: isUserScrollingRef.current,
  }
}
