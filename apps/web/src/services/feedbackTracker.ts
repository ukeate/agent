/**
 * 用户反馈追踪服务
 * 收集隐式和显式反馈事件，支持实时传输和批量处理
 */

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
  
  // 重复事件过滤
  private recentEvents = new Map<string, number>();
  private readonly DEDUP_WINDOW = 1000; // 1秒内的重复事件过滤

  constructor(config: TrackerConfig) {
    this.config = config;
    this.init();
  }

  private init(): void {
    if (this.config.enableImplicitTracking) {
      this.setupImplicitTracking();
    }
    
    this.setupNetworkMonitoring();
    this.startFlushTimer();
    
    if (this.config.websocketEndpoint) {
      this.setupWebSocket();
    }

    // 页面卸载时发送剩余事件
    window.addEventListener('beforeunload', () => {
      this.flush(true);
    });

    // 页面可见性变化时处理
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        this.onPageBlur();
      } else {
        this.onPageFocus();
      }
    });
  }

  /**
   * 设置隐式事件追踪
   */
  private setupImplicitTracking(): void {
    // 点击事件
    document.addEventListener('click', (event) => {
      this.trackClickEvent(event);
    }, { capture: true });

    // 滚动事件（节流）
    let scrollTimeout: number | null = null;
    document.addEventListener('scroll', () => {
      if (scrollTimeout) clearTimeout(scrollTimeout);
      scrollTimeout = window.setTimeout(() => {
        this.trackScrollEvent();
      }, 100);
    }, { passive: true });

    // 鼠标悬停事件（节流）
    let hoverTimeout: number | null = null;
    document.addEventListener('mouseover', (event) => {
      if (hoverTimeout) clearTimeout(hoverTimeout);
      hoverTimeout = window.setTimeout(() => {
        this.trackHoverEvent(event);
      }, 200);
    }, { passive: true });

    // 页面焦点事件
    window.addEventListener('focus', () => this.onPageFocus());
    window.addEventListener('blur', () => this.onPageBlur());

    // 开始页面视图追踪
    this.startViewTracking();
  }

  /**
   * 网络状态监控
   */
  private setupNetworkMonitoring(): void {
    window.addEventListener('online', () => {
      this.isOnline = true;
      this.processRetryQueue();
    });

    window.addEventListener('offline', () => {
      this.isOnline = false;
    });
  }

  /**
   * 设置WebSocket连接
   */
  private setupWebSocket(): void {
    if (!this.config.websocketEndpoint) return;

    try {
      this.websocket = new WebSocket(this.config.websocketEndpoint);
      
      this.websocket.onopen = () => {
        if (this.config.debug) {
          console.log('[FeedbackTracker] WebSocket连接已建立');
        }
      };

      this.websocket.onclose = () => {
        if (this.config.debug) {
          console.log('[FeedbackTracker] WebSocket连接已关闭');
        }
        // 5秒后尝试重连
        setTimeout(() => this.setupWebSocket(), 5000);
      };

      this.websocket.onerror = (error) => {
        console.error('[FeedbackTracker] WebSocket错误:', error);
      };
    } catch (error) {
      console.error('[FeedbackTracker] WebSocket初始化失败:', error);
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
      console.log('[FeedbackTracker] 事件已添加到缓冲区:', event);
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
        console.log('[FeedbackTracker] 重试队列处理成功');
      }
    } catch (error) {
      // 重新加入重试队列
      this.retryQueue.unshift(...events);
      console.error('[FeedbackTracker] 重试队列处理失败:', error);
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
        console.warn('[FeedbackTracker] WebSocket发送失败，回退到HTTP');
      }
    }

    // 回退到HTTP API
    try {
      const response = await fetch(this.config.apiEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          batch_id: this.generateEventId(),
          user_id: this.config.userId,
          session_id: this.config.sessionId,
          events,
          timestamp: Date.now()
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      if (this.config.debug) {
        console.log('[FeedbackTracker] 事件发送成功:', events.length);
      }
    } catch (error) {
      console.error('[FeedbackTracker] 事件发送失败:', error);
      this.retryQueue.push(...events);
      throw error;
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
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
    }

    if (this.websocket) {
      this.websocket.close();
    }

    // 最后一次刷新
    this.flush(true).catch(console.error);
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
  apiEndpoint: '/api/v1/feedback/implicit',
  websocketEndpoint: undefined, // 可选配置
  debug: process.env.NODE_ENV === 'development'
};

export default FeedbackTracker;