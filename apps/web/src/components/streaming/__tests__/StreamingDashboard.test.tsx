/**
 * StreamingDashboard 组件单元测试
 */

import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { StreamingDashboard } from '../StreamingDashboard';
import { streamingService } from '../../../services/streamingService';

// Mock streamingService
vi.mock('../../../services/streamingService', () => ({
  streamingService: {
    getSystemMetrics: vi.fn(),
    getBackpressureStatus: vi.fn(),
    getFlowControlMetrics: vi.fn(),
    getQueueStatus: vi.fn(),
    getSessions: vi.fn(),
    getHealth: vi.fn()
  }
}));

describe('StreamingDashboard', () => {
  const mockSystemMetrics = {
    active_sessions: 5,
    total_tokens_processed: 10000,
    tokens_per_second: 150.5,
    avg_latency_ms: 25.3,
    error_rate: 0.5,
    buffer_utilization: 0.65,
    cpu_usage: 45.2,
    memory_usage: 62.8
  };

  const mockBackpressureStatus = {
    throttle_level: 'light',
    buffer_usage: 650,
    buffer_usage_ratio: 0.65,
    pressure_metrics: {
      buffer_overflow: {
        current_value: 0.65,
        threshold: 0.8,
        severity: 0.2,
        over_threshold: false
      }
    },
    is_monitoring: true,
    active_throttles: ['rate_limiting']
  };

  const mockFlowControlMetrics = {
    rate_limiter: {
      rate: 100,
      current_allowance: 85.5,
      total_requests: 1000,
      total_allowed: 950,
      total_rejected: 50,
      rejection_rate: 5,
      avg_wait_time: 0.025
    },
    circuit_breaker: {
      state: 'CLOSED',
      failure_count: 0,
      failure_threshold: 5,
      recovery_timeout: 60
    }
  };

  const mockQueueStatus = {
    queues: {
      main: {
        name: 'main',
        depth: 150,
        max_size: 1000,
        utilization: 0.15,
        throughput: 50.5,
        avg_wait_time: 1.2,
        oldest_item_age: 5.5
      }
    }
  };

  const mockSessions = {
    sessions: {
      'session-1': {
        session_id: 'session-1',
        agent_id: 'agent-1',
        status: 'processing',
        created_at: '2025-08-15T10:00:00Z',
        token_count: 500,
        event_count: 10,
        error_count: 0,
        tokens_per_second: 50.5,
        last_event_time: '2025-08-15T10:01:00Z'
      }
    }
  };

  const mockHealthStatus = {
    status: 'healthy',
    uptime: 3600,
    version: '1.0.0'
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(streamingService.getSystemMetrics).mockResolvedValue(mockSystemMetrics);
    vi.mocked(streamingService.getBackpressureStatus).mockResolvedValue(mockBackpressureStatus);
    vi.mocked(streamingService.getFlowControlMetrics).mockResolvedValue(mockFlowControlMetrics);
    vi.mocked(streamingService.getQueueStatus).mockResolvedValue(mockQueueStatus);
    vi.mocked(streamingService.getSessions).mockResolvedValue(mockSessions);
    vi.mocked(streamingService.getHealth).mockResolvedValue(mockHealthStatus);
  });

  it('渲染监控面板并显示加载状态', () => {
    render(<StreamingDashboard />);
    expect(screen.getByText('加载中...')).toBeInTheDocument();
  });

  it('成功加载并显示系统指标', async () => {
    render(<StreamingDashboard />);

    await waitFor(() => {
      expect(screen.getByText('系统指标')).toBeInTheDocument();
      expect(screen.getByText('5')).toBeInTheDocument(); // active_sessions
      expect(screen.getByText('150.5 t/s')).toBeInTheDocument(); // tokens_per_second
      expect(screen.getByText('25.3 ms')).toBeInTheDocument(); // avg_latency_ms
      expect(screen.getByText('0.5%')).toBeInTheDocument(); // error_rate
    });
  });

  it('显示背压状态信息', async () => {
    render(<StreamingDashboard />);

    await waitFor(() => {
      expect(screen.getByText('背压控制')).toBeInTheDocument();
      expect(screen.getByText('轻度限流')).toBeInTheDocument();
      expect(screen.getByText('65.0%')).toBeInTheDocument(); // buffer usage ratio
    });
  });

  it('显示流量控制指标', async () => {
    render(<StreamingDashboard />);

    await waitFor(() => {
      expect(screen.getByText('流量控制')).toBeInTheDocument();
      expect(screen.getByText('CLOSED')).toBeInTheDocument(); // circuit breaker state
      expect(screen.getByText('5.0%')).toBeInTheDocument(); // rejection rate
    });
  });

  it('显示队列状态', async () => {
    render(<StreamingDashboard />);

    await waitFor(() => {
      expect(screen.getByText('队列健康度')).toBeInTheDocument();
      expect(screen.getByText('main')).toBeInTheDocument();
      expect(screen.getByText('15.0%')).toBeInTheDocument(); // queue utilization
    });
  });

  it('切换自动刷新', async () => {
    render(<StreamingDashboard />);

    await waitFor(() => {
      const checkbox = screen.getByRole('checkbox');
      expect(checkbox).toBeChecked();

      fireEvent.click(checkbox);
      expect(checkbox).not.toBeChecked();
    });
  });

  it('更改刷新间隔', async () => {
    render(<StreamingDashboard />);

    await waitFor(() => {
      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: '10000' } });
      expect(select).toHaveValue('10000');
    });
  });

  it('处理API错误', async () => {
    const errorMessage = '获取数据失败';
    vi.mocked(streamingService.getSystemMetrics).mockRejectedValue(new Error(errorMessage));

    render(<StreamingDashboard />);

    await waitFor(() => {
      expect(screen.getByText(errorMessage)).toBeInTheDocument();
    });
  });

  it('根据限流级别显示不同颜色', async () => {
    const testCases = [
      { level: 'none', expectedClass: 'text-green-600' },
      { level: 'light', expectedClass: 'text-yellow-500' },
      { level: 'moderate', expectedClass: 'text-orange-500' },
      { level: 'heavy', expectedClass: 'text-red-500' },
      { level: 'severe', expectedClass: 'text-red-700' }
    ];

    for (const testCase of testCases) {
      vi.mocked(streamingService.getBackpressureStatus).mockResolvedValue({
        ...mockBackpressureStatus,
        throttle_level: testCase.level
      });

      const { container } = render(<StreamingDashboard />);

      await waitFor(() => {
        const throttleLevelElement = container.querySelector(`.${testCase.expectedClass}`);
        expect(throttleLevelElement).toBeInTheDocument();
      });
    }
  });

  it('根据队列健康度显示不同颜色', async () => {
    const testCases = [
      { utilization: 0.3, expectedClass: 'text-green-600' },
      { utilization: 0.6, expectedClass: 'text-yellow-600' },
      { utilization: 0.85, expectedClass: 'text-red-600' }
    ];

    for (const testCase of testCases) {
      vi.mocked(streamingService.getQueueStatus).mockResolvedValue({
        queues: {
          main: {
            ...mockQueueStatus.queues.main,
            utilization: testCase.utilization
          }
        }
      });

      const { container } = render(<StreamingDashboard />);

      await waitFor(() => {
        const healthElement = container.querySelector(`.${testCase.expectedClass}`);
        expect(healthElement).toBeInTheDocument();
      });
    }
  });

  it('定期刷新数据', async () => {
    vi.useFakeTimers();
    render(<StreamingDashboard />);

    await waitFor(() => {
      expect(streamingService.getSystemMetrics).toHaveBeenCalledTimes(1);
    });

    vi.advanceTimersByTime(5000);

    await waitFor(() => {
      expect(streamingService.getSystemMetrics).toHaveBeenCalledTimes(2);
    });

    vi.useRealTimers();
  });
});