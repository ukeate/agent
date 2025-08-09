/**
 * FallbackHandler组件单元测试
 * 
 * 测试功能包括：
 * - 基础渲染和触发按钮
 * - 自动触发机制
 * - 故障分析和策略推荐
 * - 后备策略执行
 * - 人工协助功能
 * - 执行历史记录
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, Mock } from 'vitest';
import FallbackHandler from '../../../src/components/agentic-rag/FallbackHandler';

// Mock dependencies
const mockRagStore = {
  currentQuery: '',
  queryResults: [],
  isQuerying: false,
  error: null,
  clearErrors: vi.fn(),
};

vi.mock('../../../src/stores/ragStore', () => ({
  useRagStore: () => mockRagStore,
}));

const mockRagService = {
  query: vi.fn(),
};

vi.mock('../../../src/services/ragService', () => ({
  ragService: mockRagService,
  QueryRequest: {},
}));

// Mock message
vi.mock('antd', async () => {
  const actual = await vi.importActual('antd');
  return {
    ...actual,
    message: {
      success: vi.fn(),
      error: vi.fn(),
      info: vi.fn(),
    },
  };
});

describe('FallbackHandler', () => {
  const mockOnFallbackSuccess = vi.fn();
  const mockOnFallbackFailed = vi.fn();
  
  beforeEach(() => {
    vi.clearAllMocks();
    mockRagStore.currentQuery = '';
    mockRagStore.queryResults = [];
    mockRagStore.isQuerying = false;
    mockRagStore.error = null;
  });

  describe('基础渲染', () => {
    it('默认显示触发按钮', () => {
      render(<FallbackHandler />);
      
      expect(screen.getByRole('button', { name: /智能后备处理/ })).toBeInTheDocument();
    });

    it('触发按钮在查询中时禁用', () => {
      mockRagStore.isQuerying = true;
      
      render(<FallbackHandler />);
      
      expect(screen.getByRole('button', { name: /智能后备处理/ })).toBeDisabled();
    });

    it('触发按钮在无查询时禁用', () => {
      mockRagStore.currentQuery = '';
      
      render(<FallbackHandler />);
      
      expect(screen.getByRole('button', { name: /智能后备处理/ })).toBeDisabled();
    });
  });

  describe('自动触发机制', () => {
    it('有错误时自动触发', async () => {
      mockRagStore.currentQuery = '测试查询';
      mockRagStore.error = '查询失败';
      mockRagStore.isQuerying = false;
      
      render(<FallbackHandler autoTrigger={true} />);
      
      await waitFor(() => {
        expect(screen.getByText('智能后备处理')).toBeInTheDocument();
        expect(screen.getByText('故障分析')).toBeInTheDocument();
      });
    });

    it('无结果时自动触发', async () => {
      mockRagStore.currentQuery = '测试查询';
      mockRagStore.queryResults = [];
      mockRagStore.isQuerying = false;
      
      render(<FallbackHandler autoTrigger={true} />);
      
      await waitFor(() => {
        expect(screen.getByText('智能后备处理')).toBeInTheDocument();
      });
    });

    it('结果质量低时自动触发', async () => {
      mockRagStore.currentQuery = '测试查询';
      mockRagStore.queryResults = [
        { id: '1', content: '测试', score: 0.2 },
        { id: '2', content: '测试2', score: 0.1 },
      ];
      mockRagStore.isQuerying = false;
      
      render(<FallbackHandler autoTrigger={true} />);
      
      await waitFor(() => {
        expect(screen.getByText('智能后备处理')).toBeInTheDocument();
      });
    });

    it('不启用自动触发时不会自动显示', () => {
      mockRagStore.currentQuery = '测试查询';
      mockRagStore.error = '查询失败';
      
      render(<FallbackHandler autoTrigger={false} />);
      
      expect(screen.queryByText('故障分析')).not.toBeInTheDocument();
    });
  });

  describe('手动触发故障分析', () => {
    beforeEach(() => {
      mockRagStore.currentQuery = '如何实现多智能体协作';
    });

    it('点击按钮触发故障分析', async () => {
      render(<FallbackHandler />);
      
      const user = userEvent.setup();
      const triggerButton = screen.getByRole('button', { name: /智能后备处理/ });
      await user.click(triggerButton);
      
      expect(screen.getByText('故障分析')).toBeInTheDocument();
      expect(screen.getByText('检测到问题: 无检索结果')).toBeInTheDocument();
      expect(screen.getByText(/查询: "如何实现多智能体协作"/)).toBeInTheDocument();
    });

    it('显示推荐策略', async () => {
      render(<FallbackHandler />);
      
      const user = userEvent.setup();
      const triggerButton = screen.getByRole('button', { name: /智能后备处理/ });
      await user.click(triggerButton);
      
      expect(screen.getByText('推荐策略:')).toBeInTheDocument();
      expect(screen.getByText(/查询扩展/)).toBeInTheDocument();
      expect(screen.getByText(/查询简化/)).toBeInTheDocument();
    });

    it('可以选择和取消选择策略', async () => {
      render(<FallbackHandler />);
      
      const user = userEvent.setup();
      const triggerButton = screen.getByRole('button', { name: /智能后备处理/ });
      await user.click(triggerButton);
      
      // 查找包含"查询扩展"文本的策略标签
      const expansionStrategy = screen.getByText(/查询扩展/);
      await user.click(expansionStrategy);
      
      // 验证策略被选中（通过样式变化验证）
      expect(expansionStrategy.closest('.ant-tag')).toHaveClass('ant-tag-blue');
      
      // 再次点击取消选择
      await user.click(expansionStrategy);
      expect(expansionStrategy.closest('.ant-tag')).not.toHaveClass('ant-tag-blue');
    });
  });

  describe('后备策略执行', () => {
    beforeEach(() => {
      mockRagStore.currentQuery = '如何实现多智能体协作';
      mockRagService.query.mockResolvedValue({
        success: true,
        results: [
          { id: '1', content: '多智能体实现方法', score: 0.9 },
          { id: '2', content: '协作机制详解', score: 0.8 },
        ],
      });
    });

    it('执行成功的后备策略', async () => {
      render(
        <FallbackHandler
          onFallbackSuccess={mockOnFallbackSuccess}
        />
      );
      
      const user = userEvent.setup();
      
      // 触发故障分析
      const triggerButton = screen.getByRole('button', { name: /智能后备处理/ });
      await user.click(triggerButton);
      
      // 执行后备策略
      const executeButton = screen.getByRole('button', { name: /执行后备策略/ });
      await user.click(executeButton);
      
      // 等待执行完成
      await waitFor(() => {
        expect(screen.getByText('执行进度')).toBeInTheDocument();
        expect(screen.getByText('执行成功')).toBeInTheDocument();
      });
      
      expect(mockOnFallbackSuccess).toHaveBeenCalledWith([
        { id: '1', content: '多智能体实现方法', score: 0.9 },
        { id: '2', content: '协作机制详解', score: 0.8 },
      ]);
    });

    it('处理执行失败的后备策略', async () => {
      mockRagService.query.mockRejectedValue(new Error('服务不可用'));
      
      render(
        <FallbackHandler
          onFallbackFailed={mockOnFallbackFailed}
        />
      );
      
      const user = userEvent.setup();
      
      // 触发故障分析
      const triggerButton = screen.getByRole('button', { name: /智能后备处理/ });
      await user.click(triggerButton);
      
      // 执行后备策略
      const executeButton = screen.getByRole('button', { name: /执行后备策略/ });
      await user.click(executeButton);
      
      // 等待执行完成
      await waitFor(() => {
        expect(screen.getByText('执行失败')).toBeInTheDocument();
      });
      
      expect(mockOnFallbackFailed).toHaveBeenCalledWith('服务不可用');
    });

    it('策略执行中显示加载状态', async () => {
      // 模拟慢响应
      mockRagService.query.mockImplementation(() => 
        new Promise(resolve => setTimeout(() => resolve({
          success: true,
          results: [],
        }), 100))
      );
      
      render(<FallbackHandler />);
      
      const user = userEvent.setup();
      
      // 触发故障分析
      const triggerButton = screen.getByRole('button', { name: /智能后备处理/ });
      await user.click(triggerButton);
      
      // 执行后备策略
      const executeButton = screen.getByRole('button', { name: /执行后备策略/ });
      await user.click(executeButton);
      
      // 验证加载状态
      expect(screen.getByText('执行中...')).toBeInTheDocument();
    });
  });

  describe('人工协助功能', () => {
    beforeEach(async () => {
      render(<FallbackHandler />);
      
      const user = userEvent.setup();
      // 先触发故障分析
      mockRagStore.currentQuery = '复杂查询';
      const triggerButton = screen.getByRole('button', { name: /智能后备处理/ });
      await user.click(triggerButton);
    });

    it('打开人工协助模态框', async () => {
      const user = userEvent.setup();
      const helpButton = screen.getByRole('button', { name: /请求人工协助/ });
      
      await user.click(helpButton);
      
      expect(screen.getByText('请求人工协助')).toBeInTheDocument();
      expect(screen.getByText('我们的专家会在24小时内回复您的请求')).toBeInTheDocument();
    });

    it('提交人工协助请求', async () => {
      const user = userEvent.setup();
      const helpButton = screen.getByRole('button', { name: /请求人工协助/ });
      
      await user.click(helpButton);
      
      // 填写请求内容
      const requestTextarea = screen.getByPlaceholderText(/请描述您希望找到什么信息/);
      await user.type(requestTextarea, '我需要关于多智能体协作的详细资料');
      
      // 填写反馈内容
      const feedbackTextarea = screen.getByPlaceholderText(/对当前搜索结果有什么看法/);
      await user.type(feedbackTextarea, '当前结果不够详细');
      
      // 提交请求
      const submitButton = screen.getByRole('button', { name: '提交请求' });
      await user.click(submitButton);
      
      // 验证模态框关闭
      await waitFor(() => {
        expect(screen.queryByText('请求人工协助')).not.toBeInTheDocument();
      });
    });

    it('提交空请求时显示错误', async () => {
      const user = userEvent.setup();
      const helpButton = screen.getByRole('button', { name: /请求人工协助/ });
      
      await user.click(helpButton);
      
      // 直接提交空请求
      const submitButton = screen.getByRole('button', { name: '提交请求' });
      await user.click(submitButton);
      
      // 模态框应该仍然打开
      expect(screen.getByText('请求人工协助')).toBeInTheDocument();
    });

    it('可以取消人工协助请求', async () => {
      const user = userEvent.setup();
      const helpButton = screen.getByRole('button', { name: /请求人工协助/ });
      
      await user.click(helpButton);
      
      const cancelButton = screen.getByRole('button', { name: '取消' });
      await user.click(cancelButton);
      
      expect(screen.queryByText('请求人工协助')).not.toBeInTheDocument();
    });
  });

  describe('历史记录功能', () => {
    it('showHistory为false时不显示历史记录', () => {
      render(<FallbackHandler showHistory={false} />);
      
      expect(screen.queryByText('历史记录')).not.toBeInTheDocument();
    });

    it('执行成功后添加到历史记录', async () => {
      mockRagService.query.mockResolvedValue({
        success: true,
        results: [{ id: '1', content: '测试结果', score: 0.9 }],
      });
      
      render(<FallbackHandler showHistory={true} />);
      
      const user = userEvent.setup();
      mockRagStore.currentQuery = '测试查询';
      
      // 触发并执行策略
      const triggerButton = screen.getByRole('button', { name: /智能后备处理/ });
      await user.click(triggerButton);
      
      const executeButton = screen.getByRole('button', { name: /执行后备策略/ });
      await user.click(executeButton);
      
      // 等待执行完成后查看历史记录
      await waitFor(() => {
        expect(screen.getByText('历史记录')).toBeInTheDocument();
        expect(screen.getByText('测试查询')).toBeInTheDocument();
        expect(screen.getByText('成功')).toBeInTheDocument();
      });
    });
  });

  describe('不同故障类型的分析', () => {
    it('分析超时错误', async () => {
      mockRagStore.currentQuery = '复杂查询';
      mockRagStore.error = 'request timeout';
      
      render(<FallbackHandler />);
      
      const user = userEvent.setup();
      const triggerButton = screen.getByRole('button', { name: /智能后备处理/ });
      await user.click(triggerButton);
      
      expect(screen.getByText('检测到问题: 请求超时')).toBeInTheDocument();
      expect(screen.getByText(/网络连接问题/)).toBeInTheDocument();
    });

    it('分析服务错误', async () => {
      mockRagStore.currentQuery = '测试查询';
      mockRagStore.error = '服务器内部错误';
      
      render(<FallbackHandler />);
      
      const user = userEvent.setup();
      const triggerButton = screen.getByRole('button', { name: /智能后备处理/ });
      await user.click(triggerButton);
      
      expect(screen.getByText('检测到问题: 服务错误')).toBeInTheDocument();
      expect(screen.getByText(/服务暂时不可用/)).toBeInTheDocument();
    });

    it('分析低相关性结果', async () => {
      mockRagStore.currentQuery = '测试查询';
      mockRagStore.queryResults = [
        { id: '1', content: '低质量结果', score: 0.2 },
      ];
      
      render(<FallbackHandler />);
      
      const user = userEvent.setup();
      const triggerButton = screen.getByRole('button', { name: /智能后备处理/ });
      await user.click(triggerButton);
      
      expect(screen.getByText('检测到问题: 相关性低')).toBeInTheDocument();
      expect(screen.getByText(/查询意图理解偏差/)).toBeInTheDocument();
    });
  });

  describe('组件关闭功能', () => {
    it('可以手动关闭后备处理面板', async () => {
      render(<FallbackHandler />);
      
      const user = userEvent.setup();
      mockRagStore.currentQuery = '测试查询';
      
      // 触发故障分析
      const triggerButton = screen.getByRole('button', { name: /智能后备处理/ });
      await user.click(triggerButton);
      
      expect(screen.getByText('智能后备处理')).toBeInTheDocument();
      
      // 关闭面板
      const closeButton = screen.getByRole('button', { name: '关闭' });
      await user.click(closeButton);
      
      expect(screen.queryByText('故障分析')).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: /智能后备处理/ })).toBeInTheDocument();
    });

    it('成功执行后自动关闭', async () => {
      vi.useFakeTimers();
      
      mockRagService.query.mockResolvedValue({
        success: true,
        results: [{ id: '1', content: '成功结果', score: 0.9 }],
      });
      
      render(<FallbackHandler />);
      
      const user = userEvent.setup();
      mockRagStore.currentQuery = '测试查询';
      
      // 触发并执行策略
      const triggerButton = screen.getByRole('button', { name: /智能后备处理/ });
      await user.click(triggerButton);
      
      const executeButton = screen.getByRole('button', { name: /执行后备策略/ });
      await user.click(executeButton);
      
      // 等待执行完成
      await waitFor(() => {
        expect(screen.getByText('执行成功')).toBeInTheDocument();
      });
      
      // 快进时间，触发自动关闭
      vi.advanceTimersByTime(3000);
      
      await waitFor(() => {
        expect(screen.queryByText('故障分析')).not.toBeInTheDocument();
      });
      
      vi.useRealTimers();
    });
  });

  describe('自定义类名和回调', () => {
    it('应用自定义className', () => {
      const { container } = render(
        <FallbackHandler className="custom-fallback-handler" />
      );
      
      expect(container.querySelector('.custom-fallback-handler')).toBeInTheDocument();
    });

    it('调用成功回调函数', async () => {
      mockRagService.query.mockResolvedValue({
        success: true,
        results: [{ id: '1', content: '测试', score: 0.9 }],
      });
      
      render(
        <FallbackHandler
          onFallbackSuccess={mockOnFallbackSuccess}
          onFallbackFailed={mockOnFallbackFailed}
        />
      );
      
      const user = userEvent.setup();
      mockRagStore.currentQuery = '测试查询';
      
      // 触发并执行策略
      const triggerButton = screen.getByRole('button', { name: /智能后备处理/ });
      await user.click(triggerButton);
      
      const executeButton = screen.getByRole('button', { name: /执行后备策略/ });
      await user.click(executeButton);
      
      await waitFor(() => {
        expect(mockOnFallbackSuccess).toHaveBeenCalledWith([
          { id: '1', content: '测试', score: 0.9 }
        ]);
      });
    });
  });
});