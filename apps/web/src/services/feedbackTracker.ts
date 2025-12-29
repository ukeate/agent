/**
 * 用户反馈追踪服务
 * 收集隐式和显式反馈事件，支持实时传输和批量处理
 */

import { apiFetch, buildApiUrl } from '../utils/apiBase';

import { logger } from '../utils/logger'
// 反馈事件类型枚举
export enum FeedbackType {
  // 隐式反馈
  CLICK = 'click',
  VIEW = 'view',
  DWELL_TIME = 'dwell_time',
  SCROLL_DEPTH = 'scroll_depth',
  HOVER = 'hover',
  FOCUS = 'focus',
  BLUR = 'blur',
  
  // 显式反馈
  RATING = 'rating',
  LIKE = 'like',
  DISLIKE = 'dislike',
  BOOKMARK = 'bookmark',
  SHARE = 'share',
  COMMENT = 'comment'
}

// 反馈事件接口
export interface FeedbackEvent {
  eventId: string;
  userId: string;
  sessionId: string;
  itemId?: string;
  feedbackType: FeedbackType;
  value: number | string | boolean;
  rawValue?: any;
  context: {
    url: string;
    pageTitle: string;
    elementId?: string;
    elementType?: string;
    coordinates?: { x: number; y: number };
    viewport?: { width: number; height: number };
    timestamp: number;
    userAgent: string;
  };
  metadata?: Record<string, any>;
}

// 事件缓冲配置
interface BufferConfig {
  maxSize: number;
  flushInterval: number; // 毫秒
  maxRetries: number;
  retryDelay: number;
}

// 追踪配置
interface TrackerConfig {
  userId: string;
  sessionId: string;
  enableImplicitTracking: boolean;
  enableExplicitTracking: boolean;
  bufferConfig: BufferConfig;
  apiEndpoint: string;
  websocketEndpoint?: string;
  debug: boolean;
}

// 页面视图追踪状态
interface ViewTrackingState {
  startTime: number;
  scrollDepth: number;
  maxScrollDepth: number;
  focusTime: number;
  interactionCount: number;
}

class FeedbackTracker {
  private config: TrackerConfig;
  private eventBuffer: FeedbackEvent[] = [];
  private flushTimer: number | null = null;
  private websocket: WebSocket | null = null;
  private viewTrackingState: ViewTrackingState | null = null;
  private eventSequence = 0;
  private retryQueue: FeedbackEvent[] = [];
  private isOnline = true;
  private disposed = false;
  private shouldReconnect = true;
  private reconnectTimer: number | null = null;
  private scrollTimeout: number | null = null;
  private hoverTimeout: number | null = null;
  private flushOnHideHandler: (() => void) | null = null;
  private visibilityChangeHandler: (() => void) | null = null;
  private pageHideHandler: (() => void) | null = null;
  private clickHandler: ((event: MouseEvent) => void) | null = null;
  private scrollHandler: (() => void) | null = null;
  private hoverHandler: ((event: MouseEvent) => void) | null = null;
  private focusHandler: (() => void) | null = null;
  private blurHandler: (() => void) | null = null;
  private onlineHandler: (() => void) | null = null;
  private offlineHandler: (() => void) | null = null;
  
  // 重复事件过滤
  private recentEvents = new Map<string, number>();
  private readonly DEDUP_WINDOW = 1000; // 1秒内的重复事件过滤

  constructor(config: TrackerConfig) {
    this.config = config;
    this.init();
  }

  private init(): void {
    if (this.disposed) return;
    if (typeof window === 'undefined' || typeof document === 'undefined') return;
    if (this.config.enableImplicitTracking) {
      this.setupImplicitTracking();
    }
    
    this.setupNetworkMonitoring();
    this.startFlushTimer();
    
    if (this.config.websocketEndpoint) {
      this.setupWebSocket();
    }

    this.flushOnHideHandler = () => {
      this.flush(true).catch((error) => {
        if (this.config.debug) {
          logger.error('[FeedbackTracker] 页面隐藏时发送失败:', error);
        }
      });
    };

    // 页面可见性变化时处理
    this.visibilityChangeHandler = () => {
      if (document.hidden) {
        this.onPageBlur();
        this.flushOnHideHandler?.();
      } else {
        this.onPageFocus();
      }
    };
    document.addEventListener('visibilitychange', this.visibilityChangeHandler);

    this.pageHideHandler = () => {
      this.flushOnHideHandler?.();
    };
    window.addEventListener('pagehide', this.pageHideHandler);
  }

  /**
   * 设置隐式事件追踪
   */
  private setupImplicitTracking(): void {
    // 点击事件
    this.clickHandler = (event) => {
      this.trackClickEvent(event);
    };
    document.addEventListener('click', this.clickHandler, { capture: true });

    // 滚动事件（节流）
    this.scrollHandler = () => {
      if (this.scrollTimeout) clearTimeout(this.scrollTimeout);
      this.scrollTimeout = window.setTimeout(() => {
        this.trackScrollEvent();
      }, 100);
    };
    document.addEventListener('scroll', this.scrollHandler, { passive: true });

    // 鼠标悬停事件（节流）
    this.hoverHandler = (event) => {
      if (this.hoverTimeout) clearTimeout(this.hoverTimeout);
      this.hoverTimeout = window.setTimeout(() => {
        this.trackHoverEvent(event);
      }, 200);
    };
    document.addEventListener('mouseover', this.hoverHandler, { passive: true });

    // 页面焦点事件
    this.focusHandler = () => this.onPageFocus();
    this.blurHandler = () => this.onPageBlur();
    window.addEventListener('focus', this.focusHandler);
    window.addEventListener('blur', this.blurHandler);

    // 开始页面视图追踪
    this.startViewTracking();
  }

  /**
   * 网络状态监控
   */
  private setupNetworkMonitoring(): void {
    this.onlineHandler = () => {
      this.isOnline = true;
      this.processRetryQueue();
    };
    this.offlineHandler = () => {
      this.isOnline = false;
    };
    window.addEventListener('online', this.onlineHandler);
    window.addEventListener('offline', this.offlineHandler);
  }

  /**
   * 设置WebSocket连接
   */
  private setupWebSocket(): void {
    if (!this.config.websocketEndpoint) return;
    if (this.disposed) return;
    if (
      this.websocket &&
      (this.websocket.readyState === WebSocket.OPEN ||
        this.websocket.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }
    this.clearReconnectTimer();

    try {
      this.websocket = new WebSocket(this.config.websocketEndpoint);
      
      this.websocket.onopen = () => {
        if (this.config.debug) {
          logger.log('[FeedbackTracker] WebSocket连接已建立');
        }
      };

      this.websocket.onclose = () => {
        if (this.config.debug) {
          logger.log('[FeedbackTracker] WebSocket连接已关闭');
        }
        // 5秒后尝试重连
        if (this.shouldReconnect && !this.disposed) {
          this.reconnectTimer = window.setTimeout(() => this.setupWebSocket(), 5000);
        }
      };

      this.websocket.onerror = (error) => {
        logger.error('[FeedbackTracker] WebSocket错误:', error);
        if (
          this.websocket &&
          this.websocket.readyState !== WebSocket.CLOSING &&
          this.websocket.readyState !== WebSocket.CLOSED
        ) {
          this.websocket.close();
        }
      };
    } catch (error) {
      logger.error('[FeedbackTracker] WebSocket初始化失败:', error);
    }
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  /**
   * 生成唯一事件ID
   */
  private generateEventId(): string {
    return `${Date.now()}-${this.eventSequence++}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * 检查事件是否重复
   */
  private isDuplicateEvent(eventKey: string): boolean {
    const now = Date.now();
    const lastTime = this.recentEvents.get(eventKey);
    
    if (lastTime && now - lastTime < this.DEDUP_WINDOW) {
      return true;
    }
    
    this.recentEvents.set(eventKey, now);
    
    // 清理过期记录
    for (const [key, time] of this.recentEvents.entries()) {
      if (now - time > this.DEDUP_WINDOW * 2) {
        this.recentEvents.delete(key);
      }
    }
    
    return false;
  }

  /**
   * 创建基础事件对象
   */
  private createBaseEvent(feedbackType: FeedbackType, value: any, itemId?: string): FeedbackEvent {
    return {
      eventId: this.generateEventId(),
      userId: this.config.userId,
      sessionId: this.config.sessionId,
      itemId,
      feedbackType,
      value,
      context: {
        url: window.location.href,
        pageTitle: document.title,
        timestamp: Date.now(),
        userAgent: navigator.userAgent,
        viewport: {
          width: window.innerWidth,
          height: window.innerHeight
        }
      }
    };
  }

  /**
   * 添加事件到缓冲区
   */
  private addToBuffer(event: FeedbackEvent): void {
    // 检查重复事件
    const eventKey = `${event.feedbackType}-${event.itemId || 'global'}-${event.value}`;
    if (this.isDuplicateEvent(eventKey)) {
      return;
    }

    this.eventBuffer.push(event);

    if (this.config.debug) {
      logger.log('[FeedbackTracker] 事件已添加到缓冲区:', event);
    }

    // 检查是否需要立即发送
    if (this.eventBuffer.length >= this.config.bufferConfig.maxSize) {
      this.flush();
    }
  }

  /**
   * 追踪点击事件
   */
  private trackClickEvent(event: MouseEvent): void {
    const target = event.target as HTMLElement;
    const clickEvent = this.createBaseEvent(FeedbackType.CLICK, 1);
    
    clickEvent.context.elementId = target.id;
    clickEvent.context.elementType = target.tagName.toLowerCase();
    clickEvent.context.coordinates = { x: event.clientX, y: event.clientY };
    
    // 添加元素相关信息
    clickEvent.metadata = {
      className: target.className,
      textContent: target.textContent?.slice(0, 100),
      href: target.getAttribute('href'),
      dataAttributes: this.getDataAttributes(target)
    };

    this.addToBuffer(clickEvent);
  }

  /**
   * 追踪滚动事件
   */
  private trackScrollEvent(): void {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const windowHeight = window.innerHeight;
    const documentHeight = document.documentElement.scrollHeight;
    const scrollDepth = Math.round((scrollTop + windowHeight) / documentHeight * 100);

    if (this.viewTrackingState) {
      this.viewTrackingState.scrollDepth = scrollDepth;
      this.viewTrackingState.maxScrollDepth = Math.max(
        this.viewTrackingState.maxScrollDepth,
        scrollDepth
      );
    }

    const scrollEvent = this.createBaseEvent(FeedbackType.SCROLL_DEPTH, scrollDepth);
    scrollEvent.metadata = {
      scrollTop,
      documentHeight,
      windowHeight,
      maxScrollDepth: this.viewTrackingState?.maxScrollDepth || scrollDepth
    };

    this.addToBuffer(scrollEvent);
  }

  /**
   * 追踪悬停事件
   */
  private trackHoverEvent(event: MouseEvent): void {
    const target = event.target as HTMLElement;
    const hoverEvent = this.createBaseEvent(FeedbackType.HOVER, 1);
    
    hoverEvent.context.elementId = target.id;
    hoverEvent.context.elementType = target.tagName.toLowerCase();
    hoverEvent.context.coordinates = { x: event.clientX, y: event.clientY };

    this.addToBuffer(hoverEvent);
  }

  /**
   * 开始页面视图追踪
   */
  private startViewTracking(): void {
    this.viewTrackingState = {
      startTime: Date.now(),
      scrollDepth: 0,
      maxScrollDepth: 0,
      focusTime: 0,
      interactionCount: 0
    };

    const viewEvent = this.createBaseEvent(FeedbackType.VIEW, 1);
    this.addToBuffer(viewEvent);
  }

  /**
   * 页面获得焦点
   */
  private onPageFocus(): void {
    if (this.viewTrackingState) {
      this.viewTrackingState.focusTime = Date.now();
    }

    const focusEvent = this.createBaseEvent(FeedbackType.FOCUS, 1);
    this.addToBuffer(focusEvent);
  }

  /**
   * 页面失去焦点
   */
  private onPageBlur(): void {
    if (this.viewTrackingState && this.viewTrackingState.focusTime > 0) {
      const dwellTime = Date.now() - this.viewTrackingState.focusTime;
      const dwellEvent = this.createBaseEvent(FeedbackType.DWELL_TIME, dwellTime);
      
      dwellEvent.metadata = {
        totalScrollDepth: this.viewTrackingState.maxScrollDepth,
        interactionCount: this.viewTrackingState.interactionCount
      };

      this.addToBuffer(dwellEvent);
    }

    const blurEvent = this.createBaseEvent(FeedbackType.BLUR, 1);
    this.addToBuffer(blurEvent);
  }

  /**
   * 获取元素的data属性
   */
  private getDataAttributes(element: HTMLElement): Record<string, string> {
    const dataAttrs: Record<string, string> = {};
    for (const attr of element.attributes) {
      if (attr.name.startsWith('data-')) {
        dataAttrs[attr.name] = attr.value;
      }
    }
    return dataAttrs;
  }

  /**
   * 启动刷新定时器
   */
  private startFlushTimer(): void {
    this.flushTimer = window.setInterval(() => {
      if (this.eventBuffer.length > 0) {
        this.flush();
      }
    }, this.config.bufferConfig.flushInterval);
  }

  /**
   * 处理重试队列
   */
  private async processRetryQueue(): Promise<void> {
    if (this.retryQueue.length === 0) return;

    const events = [...this.retryQueue];
    this.retryQueue = [];

    try {
      await this.sendEvents(events);
      if (this.config.debug) {
        logger.log('[FeedbackTracker] 重试队列处理成功');
      }
    } catch (error) {
      // 重新加入重试队列
      this.retryQueue.unshift(...events);
      logger.error('[FeedbackTracker] 重试队列处理失败:', error);
    }
  }

  /**
   * 发送事件到服务器
   */
  private async sendEvents(events: FeedbackEvent[]): Promise<void> {
    if (!this.isOnline) {
      this.retryQueue.push(...events);
      return;
    }

    // 优先使用WebSocket
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      try {
        this.websocket.send(JSON.stringify({
          type: 'feedback_batch',
          events
        }));
        return;
      } catch (error) {
        logger.warn('[FeedbackTracker] WebSocket发送失败，回退到HTTP');
      }
    }

    // 回退到HTTP API
    try {
      const payload = this.buildBatchPayload(events);
      const response = await apiFetch(buildApiUrl(this.config.apiEndpoint), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (this.config.debug) {
        logger.log('[FeedbackTracker] 事件发送成功:', events.length);
      }
    } catch (error) {
      logger.error('[FeedbackTracker] 事件发送失败:', error);
      this.retryQueue.push(...events);
      throw error;
    }
  }

  private buildBatchPayload(events: FeedbackEvent[]) {
    return {
      batch_id: this.generateEventId(),
      user_id: this.config.userId,
      session_id: this.config.sessionId,
      events,
      timestamp: Date.now()
    };
  }

  private trySendBeacon(events: FeedbackEvent[]): boolean {
    if (typeof navigator === 'undefined' || !navigator.sendBeacon) return false;
    try {
      const payload = this.buildBatchPayload(events);
      const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
      return navigator.sendBeacon(buildApiUrl(this.config.apiEndpoint), blob);
    } catch (error) {
      return false;
    }
  }

  /**
   * 刷新缓冲区
   */
  public async flush(force = false): Promise<void> {
    if (this.eventBuffer.length === 0) return;

    const events = [...this.eventBuffer];
    this.eventBuffer = [];

    try {
      if (force && this.trySendBeacon(events)) {
        return;
      }
      await this.sendEvents(events);
    } catch (error) {
      if (!force) {
        // 非强制刷新时，失败的事件重新加入缓冲区
        this.eventBuffer.unshift(...events);
      }
      throw error;
    }
  }

  /**
   * 追踪显式反馈
   */
  public trackExplicitFeedback(
    feedbackType: FeedbackType,
    value: any,
    itemId?: string,
    metadata?: Record<string, any>
  ): void {
    const event = this.createBaseEvent(feedbackType, value, itemId);
    event.metadata = metadata;
    this.addToBuffer(event);
  }

  /**
   * 追踪评分
   */
  public trackRating(rating: number, itemId?: string, metadata?: Record<string, any>): void {
    this.trackExplicitFeedback(FeedbackType.RATING, rating, itemId, metadata);
  }

  /**
   * 追踪点赞/踩
   */
  public trackLike(isLike: boolean, itemId?: string, metadata?: Record<string, any>): void {
    this.trackExplicitFeedback(
      isLike ? FeedbackType.LIKE : FeedbackType.DISLIKE,
      isLike ? 1 : 0,
      itemId,
      metadata
    );
  }

  /**
   * 追踪收藏
   */
  public trackBookmark(isBookmarked: boolean, itemId?: string, metadata?: Record<string, any>): void {
    this.trackExplicitFeedback(FeedbackType.BOOKMARK, isBookmarked ? 1 : 0, itemId, metadata);
  }

  /**
   * 追踪分享
   */
  public trackShare(shareType: string, itemId?: string, metadata?: Record<string, any>): void {
    this.trackExplicitFeedback(FeedbackType.SHARE, shareType, itemId, metadata);
  }

  /**
   * 追踪评论
   */
  public trackComment(comment: string, itemId?: string, metadata?: Record<string, any>): void {
    this.trackExplicitFeedback(FeedbackType.COMMENT, comment, itemId, metadata);
  }

  /**
   * 更新配置
   */
  public updateConfig(newConfig: Partial<TrackerConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  /**
   * 获取缓冲区状态
   */
  public getBufferStatus(): {
    bufferSize: number;
    retryQueueSize: number;
    isOnline: boolean;
    websocketConnected: boolean;
  } {
    return {
      bufferSize: this.eventBuffer.length,
      retryQueueSize: this.retryQueue.length,
      isOnline: this.isOnline,
      websocketConnected: this.websocket?.readyState === WebSocket.OPEN
    };
  }

  /**
   * 销毁追踪器
   */
  public destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.shouldReconnect = false;
    this.clearReconnectTimer();

    if (this.flushTimer) {
      clearInterval(this.flushTimer);
      this.flushTimer = null;
    }

    if (this.scrollTimeout) {
      clearTimeout(this.scrollTimeout);
      this.scrollTimeout = null;
    }

    if (this.hoverTimeout) {
      clearTimeout(this.hoverTimeout);
      this.hoverTimeout = null;
    }

    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }

    if (this.visibilityChangeHandler) {
      document.removeEventListener('visibilitychange', this.visibilityChangeHandler);
      this.visibilityChangeHandler = null;
    }
    if (this.pageHideHandler) {
      window.removeEventListener('pagehide', this.pageHideHandler);
      this.pageHideHandler = null;
    }
    if (this.clickHandler) {
      document.removeEventListener('click', this.clickHandler, true);
      this.clickHandler = null;
    }
    if (this.scrollHandler) {
      document.removeEventListener('scroll', this.scrollHandler);
      this.scrollHandler = null;
    }
    if (this.hoverHandler) {
      document.removeEventListener('mouseover', this.hoverHandler);
      this.hoverHandler = null;
    }
    if (this.focusHandler) {
      window.removeEventListener('focus', this.focusHandler);
      this.focusHandler = null;
    }
    if (this.blurHandler) {
      window.removeEventListener('blur', this.blurHandler);
      this.blurHandler = null;
    }
    if (this.onlineHandler) {
      window.removeEventListener('online', this.onlineHandler);
      this.onlineHandler = null;
    }
    if (this.offlineHandler) {
      window.removeEventListener('offline', this.offlineHandler);
      this.offlineHandler = null;
    }

    // 最后一次刷新
    this.flush(true).catch((error) => {
      if (this.config.debug) {
        logger.error('[FeedbackTracker] 销毁时发送失败:', error);
      }
    });
  }
}

// 默认配置
export const DEFAULT_TRACKER_CONFIG: Omit<TrackerConfig, 'userId' | 'sessionId'> = {
  enableImplicitTracking: true,
  enableExplicitTracking: true,
  bufferConfig: {
    maxSize: 50,
    flushInterval: 5000, // 5秒
    maxRetries: 3,
    retryDelay: 1000
  },
  apiEndpoint: '/feedback/implicit',
  websocketEndpoint: undefined, // 可选配置
  debug: process.env.NODE_ENV === 'development'
};

export default FeedbackTracker;
