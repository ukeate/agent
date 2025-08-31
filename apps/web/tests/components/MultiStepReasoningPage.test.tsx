import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import MultiStepReasoningPage from '../../src/pages/MultiStepReasoningPage';
import { multiStepReasoningApi } from '../../src/services/multiStepReasoningApi';

// Mock API service
mockFn()mock('../../src/services/multiStepReasoningApi', () => ({
  multiStepReasoningApi: {
    decomposeProblem: mockFn()fn(),
    startExecution: mockFn()fn(),
    getExecutionStatus: mockFn()fn(),
    controlExecution: mockFn()fn(),
    getSystemMetrics: mockFn()fn(),
    pollExecutionStatus: mockFn()fn(),
    monitorSystemMetrics: mockFn()fn(),
    getExecutionResults: mockFn()fn(),
  }
}));

// Mock API 返回数据
const mockDecompositionResult = {
  task_dag: {
    id: 'test-dag-001',
    name: '测试任务图',
    description: '测试分解任务',
    nodes: [
      {
        id: 'task_1',
        name: '问题分析',
        description: '分析问题核心',
        task_type: 'reasoning',
        dependencies: [],
        complexity_score: 3,
        estimated_duration_minutes: 5,
        priority: 10
      },
      {
        id: 'task_2',
        name: '数据收集',
        description: '收集相关数据',
        task_type: 'tool_call',
        dependencies: ['task_1'],
        complexity_score: 4,
        estimated_duration_minutes: 8,
        priority: 8
      }
    ],
    edges: [{ from: 'task_1', to: 'task_2' }],
    parallel_groups: [['task_1'], ['task_2']],
    critical_path: ['task_1', 'task_2'],
    is_acyclic: true,
    total_nodes: 2,
    max_depth: 2
  },
  workflow_definition: {
    id: 'test-workflow-001',
    name: '测试工作流',
    description: '测试工作流定义',
    steps: [],
    execution_mode: 'parallel',
    max_parallel_steps: 3,
    metadata: {}
  },
  decomposition_metadata: {
    strategy_used: 'analysis',
    complexity_achieved: 3.5,
    total_estimated_time: 13,
    parallelization_factor: 1,
    critical_path_length: 2
  }
};

const mockExecutionResult = {
  execution_id: 'test-exec-001',
  status: 'running',
  workflow_definition_id: 'test-workflow-001',
  progress: 0,
  current_step: 'task_1',
  start_time: new Date(),
  estimated_completion: new Date(Date.now() + 5 * 60 * 1000)
};

const mockSystemMetrics = {
  active_workers: 3,
  queue_depth: 12,
  average_wait_time: 2.3,
  success_rate: 0.95,
  throughput: 8.5,
  resource_utilization: {
    cpu: 0.65,
    memory: 0.45,
    redis: 0.30,
    database: 0.25
  }
};

describe('MultiStepReasoningPage', () => {
  beforeEach(() => {
    mockFn()clearAllMocks();
    
    // 设置默认 mock 返回值
    mockFn()mocked(multiStepReasoningApi.decomposeProblem).mockResolvedValue(mockDecompositionResult);
    mockFn()mocked(multiStepReasoningApi.startExecution).mockResolvedValue(mockExecutionResult);
    mockFn()mocked(multiStepReasoningApi.getSystemMetrics).mockResolvedValue(mockSystemMetrics);
    mockFn()mocked(multiStepReasoningApi.monitorSystemMetrics).mockResolvedValue(() => {});
    mockFn()mocked(multiStepReasoningApi.pollExecutionStatus).mockImplementation(() => {});
  });

  it('应该正确渲染页面基本结构', () => {
    render(<MultiStepReasoningPage />);
    
    // 验证页面标题
    expect(screen.getByText('多步推理工作流')).toBeInTheDocument();
    expect(screen.getByText('Complex Problem → CoT Decomposition → Task DAG → Distributed Execution')).toBeInTheDocument();
    
    // 验证主要组件
    expect(screen.getByText('问题输入')).toBeInTheDocument();
    expect(screen.getByText('分解配置')).toBeInTheDocument();
    expect(screen.getByText('执行配置')).toBeInTheDocument();
    expect(screen.getByText('系统监控')).toBeInTheDocument();
    
    // 验证输入组件
    expect(screen.getByPlaceholderText('输入需要分解的复杂问题...')).toBeInTheDocument();
    expect(screen.getByText('开始分解执行')).toBeInTheDocument();
  });

  it('应该正确处理问题输入', () => {
    render(<MultiStepReasoningPage />);
    
    const textarea = screen.getByPlaceholderText('输入需要分解的复杂问题...');
    const testProblem = '如何设计一个高性能的分布式系统？';
    
    fireEvent.change(textarea, { target: { value: testProblem } });
    
    expect(textarea).toHaveValue(testProblem);
    
    // 输入内容后，开始按钮应该启用
    const startButton = screen.getByText('开始分解执行');
    expect(startButton).not.toBeDisabled();
  });

  it('应该正确处理配置选项', () => {
    render(<MultiStepReasoningPage />);
    
    // 测试分解策略选择
    const strategySelect = screen.getByDisplayValue('analysis');
    fireEvent.change(strategySelect, { target: { value: 'research' } });
    expect(strategySelect).toHaveValue('research');
    
    // 测试最大深度滑块
    const depthSlider = screen.getByDisplayValue('5');
    fireEvent.change(depthSlider, { target: { value: '7' } });
    expect(screen.getByText('最大深度: 7')).toBeInTheDocument();
    
    // 切换到执行配置
    fireEvent.click(screen.getByText('执行配置'));
    
    // 测试执行模式
    const executionModeSelect = screen.getByDisplayValue('parallel');
    fireEvent.change(executionModeSelect, { target: { value: 'sequential' } });
    expect(executionModeSelect).toHaveValue('sequential');
  });

  it('应该正确处理工作流分解和执行', async () => {
    render(<MultiStepReasoningPage />);
    
    // 输入问题
    const textarea = screen.getByPlaceholderText('输入需要分解的复杂问题...');
    fireEvent.change(textarea, { target: { value: '测试问题' } });
    
    // 点击开始分解
    const startButton = screen.getByText('开始分解执行');
    fireEvent.click(startButton);
    
    // 验证API调用
    await waitFor(() => {
      expect(multiStepReasoningApi.decomposeProblem).toHaveBeenCalledWith({
        problem_statement: '测试问题',
        strategy: 'analysis',
        max_depth: 5,
        target_complexity: 5,
        enable_branching: false
      });
    });
    
    // 验证按钮状态变化
    expect(screen.getByText('分解问题中...')).toBeInTheDocument();
    
    // 等待分解完成
    await waitFor(() => {
      expect(multiStepReasoningApi.startExecution).toHaveBeenCalled();
    });
  });

  it('应该正确显示DAG可视化', async () => {
    render(<MultiStepReasoningPage />);
    
    // 模拟完成分解，显示DAG
    const textarea = screen.getByPlaceholderText('输入需要分解的复杂问题...');
    fireEvent.change(textarea, { target: { value: '测试问题' } });
    
    const startButton = screen.getByText('开始分解执行');
    fireEvent.click(startButton);
    
    // 等待DAG显示
    await waitFor(() => {
      expect(screen.getByText('任务依赖图 (DAG)')).toBeInTheDocument();
    });
    
    // 验证任务节点显示
    await waitFor(() => {
      expect(screen.getByText('问题分析')).toBeInTheDocument();
      expect(screen.getByText('数据收集')).toBeInTheDocument();
    });
    
    // 验证DAG控制按钮
    expect(screen.getByText('全览')).toBeInTheDocument();
    expect(screen.getByText('导出')).toBeInTheDocument();
  });

  it('应该正确处理任务节点点击', async () => {
    render(<MultiStepReasoningPage />);
    
    // 先启动工作流显示DAG
    const textarea = screen.getByPlaceholderText('输入需要分解的复杂问题...');
    fireEvent.change(textarea, { target: { value: '测试问题' } });
    
    fireEvent.click(screen.getByText('开始分解执行'));
    
    // 等待DAG显示
    await waitFor(() => {
      expect(screen.getByText('任务依赖图 (DAG)')).toBeInTheDocument();
    });
    
    // 点击任务节点
    const taskNode = screen.getByText('问题分析');
    fireEvent.click(taskNode);
    
    // 验证详情面板显示
    await waitFor(() => {
      expect(screen.getByText('步骤ID')).toBeInTheDocument();
      expect(screen.getByText('类型')).toBeInTheDocument();
      expect(screen.getByText('状态')).toBeInTheDocument();
    });
  });

  it('应该正确显示系统监控指标', async () => {
    render(<MultiStepReasoningPage />);
    
    // 等待系统监控数据加载
    await waitFor(() => {
      expect(screen.getByText('3')).toBeInTheDocument(); // 活跃工作器
      expect(screen.getByText('12')).toBeInTheDocument(); // 队列任务
      expect(screen.getByText('2.3s')).toBeInTheDocument(); // 平均等待
      expect(screen.getByText('95%')).toBeInTheDocument(); // 成功率
    });
    
    // 验证监控API被调用
    expect(multiStepReasoningApi.monitorSystemMetrics).toHaveBeenCalled();
  });

  it('应该正确处理执行控制', async () => {
    render(<MultiStepReasoningPage />);
    
    // 启动工作流
    const textarea = screen.getByPlaceholderText('输入需要分解的复杂问题...');
    fireEvent.change(textarea, { target: { value: '测试问题' } });
    
    fireEvent.click(screen.getByText('开始分解执行'));
    
    // 等待执行控制面板显示
    await waitFor(() => {
      expect(screen.getByText('执行控制')).toBeInTheDocument();
    });
    
    // 模拟暂停操作
    const pauseButton = screen.getAllByRole('button').find(btn => 
      btn.querySelector('svg') && btn.getAttribute('title')?.includes('暂停')
    );
    
    if (pauseButton) {
      fireEvent.click(pauseButton);
      
      // 验证控制API被调用
      await waitFor(() => {
        expect(multiStepReasoningApi.controlExecution).toHaveBeenCalledWith({
          execution_id: mockExecutionResult.execution_id,
          action: 'pause'
        });
      });
    }
  });

  it('应该正确处理错误状态', async () => {
    // Mock API 错误
    mockFn()mocked(multiStepReasoningApi.decomposeProblem).mockRejectedValue(new Error('API错误'));
    
    render(<MultiStepReasoningPage />);
    
    const textarea = screen.getByPlaceholderText('输入需要分解的复杂问题...');
    fireEvent.change(textarea, { target: { value: '测试问题' } });
    
    const startButton = screen.getByText('开始分解执行');
    fireEvent.click(startButton);
    
    // 等待错误显示
    await waitFor(() => {
      expect(screen.getByText('问题分解失败，请检查输入并重试')).toBeInTheDocument();
    });
  });

  it('应该正确处理空输入验证', () => {
    render(<MultiStepReasoningPage />);
    
    const startButton = screen.getByText('开始分解执行');
    
    // 空输入时按钮应该禁用
    expect(startButton).toBeDisabled();
    
    // 输入空白字符
    const textarea = screen.getByPlaceholderText('输入需要分解的复杂问题...');
    fireEvent.change(textarea, { target: { value: '   ' } });
    
    // 按钮仍应该禁用
    expect(startButton).toBeDisabled();
    
    // 输入有效内容
    fireEvent.change(textarea, { target: { value: '有效问题' } });
    
    // 按钮应该启用
    expect(startButton).not.toBeDisabled();
  });

  it('应该正确处理配置状态保持', () => {
    render(<MultiStepReasoningPage />);
    
    // 设置分解配置
    const strategySelect = screen.getByDisplayValue('analysis');
    fireEvent.change(strategySelect, { target: { value: 'optimization' } });
    
    const depthSlider = screen.getByDisplayValue('5');
    fireEvent.change(depthSlider, { target: { value: '8' } });
    
    // 切换到执行配置选项卡
    fireEvent.click(screen.getByText('执行配置'));
    
    // 再切换回分解配置
    fireEvent.click(screen.getByText('分解配置'));
    
    // 验证配置保持
    expect(screen.getByDisplayValue('optimization')).toBeInTheDocument();
    expect(screen.getByText('最大深度: 8')).toBeInTheDocument();
  });

  it('应该正确处理组件卸载清理', () => {
    const { unmount } = render(<MultiStepReasoningPage />);
    
    // 卸载组件
    unmount();
    
    // 验证没有内存泄漏警告
    // 在真实测试中，这里可以检查 useEffect 清理函数的调用
  });
});