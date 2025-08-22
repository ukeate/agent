import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import SchedulingMonitor from '../SchedulingMonitor';

// Mock fetch
global.fetch = jest.fn();

const mockSchedulingStats = {
  workers: [
    {
      worker_id: 'worker-1',
      current_load: 0.6,
      task_completion_rate: 0.95,
      average_task_time: 2.5,
      task_type_performance: {
        'type-a': 2.0,
        'type-b': 3.0
      },
      status: 'active'
    },
    {
      worker_id: 'worker-2',
      current_load: 0.3,
      task_completion_rate: 0.88,
      average_task_time: 3.2,
      task_type_performance: {
        'type-a': 2.8,
        'type-c': 4.1
      },
      status: 'idle'
    },
    {
      worker_id: 'worker-3',
      current_load: 0.9,
      task_completion_rate: 0.75,
      average_task_time: 4.8,
      task_type_performance: {},
      status: 'overloaded'
    }
  ],
  sla_requirements: [
    {
      id: 'sla-1',
      name: 'Critical Tasks SLA',
      target_completion_time: 5.0,
      max_failure_rate: 0.05,
      priority_weight: 2.0,
      current_performance: {
        avg_completion_time: 3.2,
        failure_rate: 0.02,
        violation_count: 0
      },
      status: 'met'
    },
    {
      id: 'sla-2',
      name: 'Standard Tasks SLA',
      target_completion_time: 10.0,
      max_failure_rate: 0.1,
      priority_weight: 1.0,
      current_performance: {
        avg_completion_time: 12.5,
        failure_rate: 0.08,
        violation_count: 3
      },
      status: 'violated'
    }
  ],
  system_resources: {
    cpu_usage: 65.5,
    memory_usage: 42.3,
    io_utilization: 78.2,
    network_usage: 23.1,
    active_connections: 45,
    queue_depth: 23
  },
  predictive_scheduling: {
    predicted_completion_times: {
      'task-1': 2.5,
      'task-2': 4.1
    },
    scaling_recommendations: {
      action: 'scale_up',
      target_workers: 5,
      confidence: 0.85,
      reason: 'High queue depth and increasing load'
    },
    resource_forecast: {
      next_hour: {
        cpu_usage: 70.0,
        memory_usage: 48.0,
        io_utilization: 82.0,
        network_usage: 28.0,
        active_connections: 52,
        queue_depth: 30
      },
      next_24h: {
        cpu_usage: 68.0,
        memory_usage: 45.0,
        io_utilization: 75.0,
        network_usage: 25.0,
        active_connections: 48,
        queue_depth: 25
      }
    }
  },
  total_tasks_scheduled: 1250,
  load_balancing_efficiency: 87.5,
  sla_compliance_rate: 92.3
};

describe('SchedulingMonitor', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockSchedulingStats
    });
  });

  it('渲染基本组件结构', async () => {
    render(<SchedulingMonitor />);
    
    await waitFor(() => {
      // 检查统计卡片
      expect(screen.getByText('调度任务总数')).toBeInTheDocument();
      expect(screen.getByText('负载均衡效率')).toBeInTheDocument();
      expect(screen.getByText('SLA合规率')).toBeInTheDocument();
      expect(screen.getByText('活跃工作者')).toBeInTheDocument();
    });
  });

  it('正确显示系统概览统计', async () => {
    render(<SchedulingMonitor />);
    
    await waitFor(() => {
      // 检查统计数值
      expect(screen.getByText('1250')).toBeInTheDocument(); // 调度任务总数
      expect(screen.getByText('87.5')).toBeInTheDocument(); // 负载均衡效率
      expect(screen.getByText('92.3')).toBeInTheDocument(); // SLA合规率
      expect(screen.getByText('1')).toBeInTheDocument(); // 活跃工作者数量
      expect(screen.getByText('/ 3')).toBeInTheDocument(); // 总工作者数量
    });
  });

  it('显示系统资源使用率', async () => {
    render(<SchedulingMonitor />);
    
    await waitFor(() => {
      expect(screen.getByText('系统资源使用率')).toBeInTheDocument();
      
      // 检查各项资源指标
      expect(screen.getByText('CPU')).toBeInTheDocument();
      expect(screen.getByText('65.5%')).toBeInTheDocument();
      
      expect(screen.getByText('内存')).toBeInTheDocument();
      expect(screen.getByText('42.3%')).toBeInTheDocument();
      
      expect(screen.getByText('I/O')).toBeInTheDocument();
      expect(screen.getByText('78.2%')).toBeInTheDocument();
      
      expect(screen.getByText('网络')).toBeInTheDocument();
      expect(screen.getByText('23.1%')).toBeInTheDocument();
      
      expect(screen.getByText('活跃连接')).toBeInTheDocument();
      expect(screen.getByText('45')).toBeInTheDocument();
      
      expect(screen.getByText('队列深度')).toBeInTheDocument();
      expect(screen.getByText('23')).toBeInTheDocument();
    });
  });

  it('显示预测性调度建议', async () => {
    render(<SchedulingMonitor />);
    
    await waitFor(() => {
      expect(screen.getByText('预测性调度建议')).toBeInTheDocument();
      expect(screen.getByText('建议扩容')).toBeInTheDocument();
      expect(screen.getByText('目标工作者数: 5')).toBeInTheDocument();
      expect(screen.getByText('置信度: 85.0%')).toBeInTheDocument();
      expect(screen.getByText('High queue depth and increasing load')).toBeInTheDocument();
    });
  });

  it('正确显示不同的扩缩容建议', async () => {
    // 测试缩容建议
    const scaleDownStats = {
      ...mockSchedulingStats,
      predictive_scheduling: {
        ...mockSchedulingStats.predictive_scheduling,
        scaling_recommendations: {
          action: 'scale_down',
          target_workers: 2,
          confidence: 0.75,
          reason: 'Low utilization detected'
        }
      }
    };

    (fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => scaleDownStats
    });

    render(<SchedulingMonitor />);
    
    await waitFor(() => {
      expect(screen.getByText('建议缩容')).toBeInTheDocument();
    });

    // 测试保持当前规模
    const maintainStats = {
      ...mockSchedulingStats,
      predictive_scheduling: {
        ...mockSchedulingStats.predictive_scheduling,
        scaling_recommendations: {
          action: 'maintain',
          target_workers: 3,
          confidence: 0.90,
          reason: 'Optimal resource utilization'
        }
      }
    };

    (fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => maintainStats
    });

    render(<SchedulingMonitor />);
    
    await waitFor(() => {
      expect(screen.getByText('保持当前规模')).toBeInTheDocument();
    });
  });

  it('显示工作者状态表格', async () => {
    render(<SchedulingMonitor />);
    
    await waitFor(() => {
      // 检查表格标题
      expect(screen.getByText('工作者状态')).toBeInTheDocument();
      
      // 检查表格列标题
      expect(screen.getByText('工作者ID')).toBeInTheDocument();
      expect(screen.getByText('状态')).toBeInTheDocument();
      expect(screen.getByText('当前负载')).toBeInTheDocument();
      expect(screen.getByText('完成率')).toBeInTheDocument();
      expect(screen.getByText('平均耗时')).toBeInTheDocument();
      
      // 检查工作者数据
      expect(screen.getByText('worker-1')).toBeInTheDocument();
      expect(screen.getByText('worker-2')).toBeInTheDocument();
      expect(screen.getByText('worker-3')).toBeInTheDocument();
      
      // 检查状态标签
      expect(screen.getByText('ACTIVE')).toBeInTheDocument();
      expect(screen.getByText('IDLE')).toBeInTheDocument();
      expect(screen.getByText('OVERLOADED')).toBeInTheDocument();
    });
  });

  it('正确显示工作者负载进度条', async () => {
    render(<SchedulingMonitor />);
    
    await waitFor(() => {
      // 检查进度条（60%, 30%, 90%）
      const progressBars = screen.getAllByRole('progressbar');
      expect(progressBars).toHaveLength(3);
      
      // worker-1: 60%
      expect(progressBars[0]).toHaveAttribute('aria-valuenow', '60');
      // worker-2: 30%  
      expect(progressBars[1]).toHaveAttribute('aria-valuenow', '30');
      // worker-3: 90%
      expect(progressBars[2]).toHaveAttribute('aria-valuenow', '90');
    });
  });

  it('正确显示完成率和平均耗时', async () => {
    render(<SchedulingMonitor />);
    
    await waitFor(() => {
      // 检查完成率（百分比格式）
      expect(screen.getByText('95.0%')).toBeInTheDocument(); // worker-1
      expect(screen.getByText('88.0%')).toBeInTheDocument(); // worker-2
      expect(screen.getByText('75.0%')).toBeInTheDocument(); // worker-3
      
      // 检查平均耗时（秒格式）
      expect(screen.getByText('2.50s')).toBeInTheDocument(); // worker-1
      expect(screen.getByText('3.20s')).toBeInTheDocument(); // worker-2
      expect(screen.getByText('4.80s')).toBeInTheDocument(); // worker-3
    });
  });

  it('显示SLA监控表格', async () => {
    render(<SchedulingMonitor />);
    
    await waitFor(() => {
      expect(screen.getByText('SLA监控')).toBeInTheDocument();
      
      // 检查SLA数据
      expect(screen.getByText('Critical Tasks SLA')).toBeInTheDocument();
      expect(screen.getByText('Standard Tasks SLA')).toBeInTheDocument();
      
      // 检查SLA状态
      expect(screen.getByText('MET')).toBeInTheDocument();
      expect(screen.getByText('VIOLATED')).toBeInTheDocument();
      
      // 检查目标完成时间
      expect(screen.getByText('5s')).toBeInTheDocument();
      expect(screen.getByText('10s')).toBeInTheDocument();
      
      // 检查当前性能数据
      expect(screen.getByText('完成时间: 3.20s')).toBeInTheDocument();
      expect(screen.getByText('失败率: 2.0%')).toBeInTheDocument();
      expect(screen.getByText('违规次数: 0')).toBeInTheDocument();
      
      expect(screen.getByText('完成时间: 12.50s')).toBeInTheDocument();
      expect(screen.getByText('失败率: 8.0%')).toBeInTheDocument();
      expect(screen.getByText('违规次数: 3')).toBeInTheDocument();
    });
  });

  it('正确显示资源使用率颜色', async () => {
    render(<SchedulingMonitor />);
    
    await waitFor(() => {
      // 检查不同使用率的进度条颜色
      const progressElements = document.querySelectorAll('.ant-progress-line');
      expect(progressElements.length).toBeGreaterThan(0);
    });
  });

  it('处理刷新操作', async () => {
    render(<SchedulingMonitor />);
    
    await waitFor(() => {
      expect(screen.getByText('刷新')).toBeInTheDocument();
    });

    // 点击刷新按钮
    const refreshButton = screen.getByText('刷新');
    fireEvent.click(refreshButton);

    // 验证API调用
    expect(fetch).toHaveBeenCalledWith('/api/v1/batch/scheduling/stats');
  });

  it('处理API错误', async () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
    (fetch as jest.Mock).mockRejectedValue(new Error('API Error'));

    render(<SchedulingMonitor />);
    
    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith(
        '获取调度统计失败:',
        expect.any(Error)
      );
    });

    consoleSpy.mockRestore();
  });

  it('定期自动刷新数据', async () => {
    jest.useFakeTimers();
    
    render(<SchedulingMonitor />);
    
    // 初始调用
    expect(fetch).toHaveBeenCalledTimes(1);
    
    // 快进10秒
    jest.advanceTimersByTime(10000);
    
    await waitFor(() => {
      // 应该再次调用API
      expect(fetch).toHaveBeenCalledTimes(2);
    });
    
    jest.useRealTimers();
  });

  it('显示更新时间', async () => {
    jest.useFakeTimers();
    const fixedDate = new Date('2024-01-01T12:00:00Z');
    jest.setSystemTime(fixedDate);

    render(<SchedulingMonitor />);
    
    await waitFor(() => {
      expect(screen.getByText(/最后更新:/)).toBeInTheDocument();
    });
    
    jest.useRealTimers();
  });

  it('显示SLA合规率颜色', async () => {
    render(<SchedulingMonitor />);
    
    await waitFor(() => {
      // SLA合规率92.3%应该显示为红色（< 95%）
      const slaStatistic = screen.getByText('92.3');
      expect(slaStatistic.closest('.ant-statistic-content-value')).toHaveStyle('color: rgb(255, 77, 79)');
    });

    // 测试高合规率
    const highComplianceStats = {
      ...mockSchedulingStats,
      sla_compliance_rate: 96.5
    };

    (fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => highComplianceStats
    });

    render(<SchedulingMonitor />);
    
    await waitFor(() => {
      const slaStatistic = screen.getByText('96.5');
      expect(slaStatistic.closest('.ant-statistic-content-value')).toHaveStyle('color: rgb(82, 196, 26)');
    });
  });

  it('显示空状态', async () => {
    const emptyStats = {
      workers: [],
      sla_requirements: [],
      system_resources: {
        cpu_usage: 0,
        memory_usage: 0,
        io_utilization: 0,
        network_usage: 0,
        active_connections: 0,
        queue_depth: 0
      },
      predictive_scheduling: {
        predicted_completion_times: {},
        scaling_recommendations: {
          action: 'maintain',
          target_workers: 0,
          confidence: 0,
          reason: 'No data available'
        },
        resource_forecast: {
          next_hour: {
            cpu_usage: 0,
            memory_usage: 0,
            io_utilization: 0,
            network_usage: 0,
            active_connections: 0,
            queue_depth: 0
          },
          next_24h: {
            cpu_usage: 0,
            memory_usage: 0,
            io_utilization: 0,
            network_usage: 0,
            active_connections: 0,
            queue_depth: 0
          }
        }
      },
      total_tasks_scheduled: 0,
      load_balancing_efficiency: 0,
      sla_compliance_rate: 100
    };

    (fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => emptyStats
    });

    render(<SchedulingMonitor />);
    
    await waitFor(() => {
      expect(screen.getByText('0')).toBeInTheDocument(); // 调度任务总数
      expect(screen.getByText('0')).toBeInTheDocument(); // 负载均衡效率
      expect(screen.getByText('100')).toBeInTheDocument(); // SLA合规率
    });
  });
});