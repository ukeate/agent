import { useRef, useEffect, useCallback } from 'react'

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
}

export const useSmartAutoScroll = ({
  messages,
  enabled = true,
  threshold = 100,
  behavior = 'smooth',
}: UseSmartAutoScrollOptions) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const isUserScrollingRef = useRef(false)
  const lastScrollTopRef = useRef(0)
  const scrollTimeoutRef = useRef<ReturnType<typeof setTimeout>>()

  // 检查是否在底部附近
  const isNearBottom = useCallback(() => {
    if (!containerRef.current) return false
    
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current
    return scrollHeight - scrollTop - clientHeight <= threshold
  }, [threshold])

  // 滚动到底部
  const scrollToBottom = useCallback(() => {
    if (!containerRef.current) return
    
    containerRef.current.scrollTo({
      top: containerRef.current.scrollHeight,
      behavior,
    })
  }, [behavior])

  // 处理用户手动滚动
  const handleScroll = useCallback(() => {
    if (!containerRef.current) return
    
    const currentScrollTop = containerRef.current.scrollTop
    
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
  }, [])

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
      }
    }, 50)

    return () => clearTimeout(scrollTimeout)
  }, [messages.length, enabled, isNearBottom, scrollToBottom])

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
    }
  }, [handleScroll])

  // 初始滚动到底部
  useEffect(() => {
    if (messages.length > 0) {
      const initialScrollTimeout = setTimeout(() => {
        scrollToBottom()
      }, 100)
      
      return () => clearTimeout(initialScrollTimeout)
    }
  }, [messages.length > 0, scrollToBottom])

  return {
    containerRef,
    scrollToBottom,
    isNearBottom,
    isUserScrolling: isUserScrollingRef.current,
  }
}