/**
 * 工作流WebSocket服务
 * 处理工作流实时状态更新
 */

interface WorkflowUpdateCallback {
  (data: any): void;
}

interface WorkflowWebSocketMessage {
  type: 'initial_status' | 'status_update' | 'pong' | 'error';
  data?: any;
}

class WorkflowWebSocketService {
  private connections: Map<string, WebSocket> = new Map();
  private callbacks: Map<string, Set<WorkflowUpdateCallback>> = new Map();
  private reconnectAttempts: Map<string, number> = new Map();
  private maxReconnectAttempts = 5;
  private reconnectDelay = 3000; // 3秒

  /**
   * 连接到指定工作流的WebSocket
   */
  connect(workflowId: string, callback: WorkflowUpdateCallback): void {
    // 如果已经有连接，直接添加回调
    if (this.connections.has(workflowId)) {
      const callbacks = this.callbacks.get(workflowId) || new Set();
      callbacks.add(callback);
      this.callbacks.set(workflowId, callbacks);
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/v1/workflows/${workflowId}/ws`;
    
    console.log(`连接到工作流WebSocket: ${wsUrl}`);
    
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log(`工作流 ${workflowId} WebSocket连接已建立`);
      this.connections.set(workflowId, ws);
      this.reconnectAttempts.set(workflowId, 0);
      
      // 添加回调
      const callbacks = this.callbacks.get(workflowId) || new Set();
      callbacks.add(callback);
      this.callbacks.set(workflowId, callbacks);
      
      // 发送ping保持连接
      this.startHeartbeat(workflowId);
    };
    
    ws.onmessage = (event) => {
      try {
        const message: WorkflowWebSocketMessage = JSON.parse(event.data);
        console.log(`收到工作流 ${workflowId} 更新:`, message);
        
        // 通知所有回调
        const callbacks = this.callbacks.get(workflowId);
        if (callbacks) {
          callbacks.forEach(cb => cb(message));
        }
      } catch (error) {
        console.error('解析WebSocket消息失败:', error);
      }
    };
    
    ws.onclose = (event) => {
      console.log(`工作流 ${workflowId} WebSocket连接已关闭:`, event.code, event.reason);
      this.handleDisconnection(workflowId);
    };
    
    ws.onerror = (error) => {
      console.error(`工作流 ${workflowId} WebSocket错误:`, error);
    };
  }

  /**
   * 断开指定工作流的WebSocket连接
   */
  disconnect(workflowId: string, callback?: WorkflowUpdateCallback): void {
    if (callback) {
      // 只移除特定回调
      const callbacks = this.callbacks.get(workflowId);
      if (callbacks) {
        callbacks.delete(callback);
        if (callbacks.size === 0) {
          // 如果没有剩余回调，关闭连接
          this.closeConnection(workflowId);
        }
      }
    } else {
      // 关闭整个连接
      this.closeConnection(workflowId);
    }
  }

  /**
   * 请求工作流状态更新
   */
  requestStatusUpdate(workflowId: string): void {
    const ws = this.connections.get(workflowId);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'get_status' }));
    }
  }

  /**
   * 关闭连接
   */
  private closeConnection(workflowId: string): void {
    const ws = this.connections.get(workflowId);
    if (ws) {
      ws.close();
      this.connections.delete(workflowId);
    }
    this.callbacks.delete(workflowId);
    this.reconnectAttempts.delete(workflowId);
  }

  /**
   * 处理连接断开
   */
  private handleDisconnection(workflowId: string): void {
    this.connections.delete(workflowId);
    
    // 如果还有回调，尝试重连
    const callbacks = this.callbacks.get(workflowId);
    if (callbacks && callbacks.size > 0) {
      this.attemptReconnect(workflowId);
    }
  }

  /**
   * 尝试重连
   */
  private attemptReconnect(workflowId: string): void {
    const attempts = this.reconnectAttempts.get(workflowId) || 0;
    
    if (attempts < this.maxReconnectAttempts) {
      this.reconnectAttempts.set(workflowId, attempts + 1);
      
      console.log(`工作流 ${workflowId} 第 ${attempts + 1} 次重连尝试`);
      
      setTimeout(() => {
        const callbacks = this.callbacks.get(workflowId);
        if (callbacks && callbacks.size > 0) {
          // 重新连接，使用第一个回调
          const firstCallback = callbacks.values().next().value;
          if (firstCallback) {
            this.connect(workflowId, firstCallback);
          }
        }
      }, this.reconnectDelay * (attempts + 1)); // 指数退避
    } else {
      console.error(`工作流 ${workflowId} 重连失败，已达到最大重试次数`);
      this.callbacks.delete(workflowId);
      this.reconnectAttempts.delete(workflowId);
    }
  }

  /**
   * 启动心跳检测
   */
  private startHeartbeat(workflowId: string): void {
    const interval = setInterval(() => {
      const ws = this.connections.get(workflowId);
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }));
      } else {
        clearInterval(interval);
      }
    }, 30000); // 30秒心跳
  }

  /**
   * 断开所有连接
   */
  disconnectAll(): void {
    for (const workflowId of this.connections.keys()) {
      this.closeConnection(workflowId);
    }
  }
}

// 导出单例实例
export const workflowWebSocketService = new WorkflowWebSocketService();

// 页面卸载时清理所有连接
window.addEventListener('beforeunload', () => {
  workflowWebSocketService.disconnectAll();
});

export default workflowWebSocketService;