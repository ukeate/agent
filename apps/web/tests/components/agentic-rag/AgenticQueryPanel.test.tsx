/**
 * AgenticQueryPanel 组件单元测试
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, Mock } from 'vitest';
import { message } from 'antd';
import AgenticQueryPanel from '../../../src/components/agentic-rag/AgenticQueryPanel';
import { ragService, AgenticQueryRequest } from '../../../src/services/ragService';
import { useRagStore } from '../../../src/stores/ragStore';

// Mock dependencies
vi.mock('../../../src/services/ragService', () => ({
  ragService: {
    agenticQuery: vi.fn(),
  },
}));

vi.mock('../../../src/stores/ragStore');

vi.mock('antd', async () => {
  const actual = await vi.importActual('antd');
  return {
    ...actual,
    message: {
      success: vi.fn(),
      error: vi.fn(),
      warning: vi.fn(),
    },
  };
});

describe('AgenticQueryPanel', () => {
  // Mock store state
  const mockStoreState = {
    currentQuery: '',
    setCurrentQuery: vi.fn(),
    isAgenticQuerying: false,
    setIsAgenticQuerying: vi.fn(),
    setQueryAnalysis: vi.fn(),
    setExpandedQueries: vi.fn(),
    currentSession: {
      id: 'session-1',
      name: '测试会话',
      context_history: ['测试历史'],
    },
    createSession: vi.fn(),
    addToSessionHistory: vi.fn(),
    setError: vi.fn(),
    clearErrors: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    (useRagStore as unknown as Mock).mockReturnValue(mockStoreState);
    (ragService.agenticQuery as Mock).mockResolvedValue({
      success: true,
      analysis_info: null,
      expanded_queries: [],
    });
  });

  afterEach(() => {
    vi.clearAllTimers();
  });

  describe('基础渲染', () => {
    it('应该正确渲染主要组件', () => {
      render(<AgenticQueryPanel />);
      
      expect(screen.getByText('智能查询')).toBeInTheDocument();
      expect(screen.getByPlaceholderText(/请输入您的查询问题/)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /智能检索/ })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /优化查询/ })).toBeInTheDocument();
    });

    it('应该显示快速查询按钮', () => {
      render(<AgenticQueryPanel />);
      
      expect(screen.getByText('快速查询')).toBeInTheDocument();
      expect(screen.getByText('如何实现多智能体协作?')).toBeInTheDocument();
      expect(screen.getByText('什么是RAG检索增强生成?')).toBeInTheDocument();
    });

    it('应该在禁用状态下禁用输入和按钮', () => {
      render(<AgenticQueryPanel disabled />);
      
      const textarea = screen.getByPlaceholderText(/请输入您的查询问题/);
      const searchButton = screen.getByRole('button', { name: /智能检索/ });
      const optimizeButton = screen.getByRole('button', { name: /优化查询/ });

      expect(textarea).toBeDisabled();
      expect(searchButton).toBeDisabled();
      expect(optimizeButton).toBeDisabled();
    });
  });

  describe('查询输入和分析', () => {
    it('应该处理查询文本输入', async () => {
      const user = userEvent.setup();
      render(<AgenticQueryPanel />);
      
      const textarea = screen.getByPlaceholderText(/请输入您的查询问题/);
      await user.type(textarea, '如何实现智能检索系统');
      
      expect(mockStoreState.setCurrentQuery).toHaveBeenCalledWith('如何实现智能检索系统');
    });

    it('应该在自动分析模式下分析查询', async () => {
      vi.useFakeTimers();
      const user = userEvent.setup();
      render(<AgenticQueryPanel autoAnalyze />);
      
      const textarea = screen.getByPlaceholderText(/请输入您的查询问题/);
      await user.type(textarea, '代码实现');
      
      // 等待防抖
      vi.advanceTimersByTime(1000);
      
      // 等待分析完成
      await waitFor(() => {
        expect(screen.getByText('查询意图:')).toBeInTheDocument();
      });

      vi.useRealTimers();
    });

    it('应该显示查询分析结果', async () => {
      vi.useFakeTimers();
      const user = userEvent.setup();
      render(<AgenticQueryPanel autoAnalyze />);
      
      const textarea = screen.getByPlaceholderText(/请输入您的查询问题/);
      await user.type(textarea, '如何实现代码功能');
      
      vi.advanceTimersByTime(1800); // 防抖时间 + 分析时间
      
      await waitFor(() => {
        expect(screen.getByText('查询意图:')).toBeInTheDocument();
        expect(screen.getByText(/复杂度:/)).toBeInTheDocument();
      });

      vi.useRealTimers();
    });
  });

  describe('查询搜索', () => {
    it('应该在空查询时显示警告', async () => {
      const user = userEvent.setup();
      render(<AgenticQueryPanel />);
      
      const searchButton = screen.getByRole('button', { name: /智能检索/ });
      await user.click(searchButton);
      
      expect(message.warning).toHaveBeenCalledWith('请输入查询内容');
    });

    it('应该成功执行查询搜索', async () => {
      const user = userEvent.setup();
      const onSearch = vi.fn();
      const onResults = vi.fn();
      
      render(
        <AgenticQueryPanel 
          onSearch={onSearch}
          onResults={onResults}
        />
      );
      
      const textarea = screen.getByPlaceholderText(/请输入您的查询问题/);
      await user.type(textarea, '测试查询');
      
      const searchButton = screen.getByRole('button', { name: /智能检索/ });
      await user.click(searchButton);
      
      expect(mockStoreState.clearErrors).toHaveBeenCalled();
      expect(mockStoreState.setIsAgenticQuerying).toHaveBeenCalledWith(true);
      
      await waitFor(() => {
        expect(onSearch).toHaveBeenCalledWith(
          expect.objectContaining({
            query: '测试查询',
            retrieval_strategies: ['semantic', 'keyword'],
            max_results: 10,
            include_explanation: true,
          })
        );
      });
    });

    it('应该处理查询搜索错误', async () => {
      const user = userEvent.setup();
      (ragService.agenticQuery as Mock).mockRejectedValue(new Error('网络错误'));
      
      render(<AgenticQueryPanel />);
      
      const textarea = screen.getByPlaceholderText(/请输入您的查询问题/);
      await user.type(textarea, '测试查询');
      
      const searchButton = screen.getByRole('button', { name: /智能检索/ });
      await user.click(searchButton);
      
      await waitFor(() => {
        expect(mockStoreState.setError).toHaveBeenCalledWith('网络错误');
        expect(message.error).toHaveBeenCalledWith('网络错误');
      });
    });

    it('应该在没有会话时创建新会话', async () => {
      const user = userEvent.setup();
      const storeWithoutSession = {
        ...mockStoreState,
        currentSession: null,
      };
      (useRagStore as unknown as Mock).mockReturnValue(storeWithoutSession);
      
      render(<AgenticQueryPanel />);
      
      const textarea = screen.getByPlaceholderText(/请输入您的查询问题/);
      await user.type(textarea, '测试查询');
      
      const searchButton = screen.getByRole('button', { name: /智能检索/ });
      await user.click(searchButton);
      
      expect(storeWithoutSession.createSession).toHaveBeenCalledWith('智能检索会话');
    });
  });

  describe('查询优化', () => {
    it('应该执行查询优化', async () => {
      const user = userEvent.setup();
      render(<AgenticQueryPanel />);
      
      const textarea = screen.getByPlaceholderText(/请输入您的查询问题/);
      await user.type(textarea, '测试优化');
      
      const optimizeButton = screen.getByRole('button', { name: /优化查询/ });
      await user.click(optimizeButton);
      
      await waitFor(() => {
        expect(screen.getByText('查询优化建议')).toBeInTheDocument();
        expect(screen.getByText(/优化后查询:/)).toBeInTheDocument();
      });
    });

    it('应该应用优化建议', async () => {
      const user = userEvent.setup();
      render(<AgenticQueryPanel />);
      
      const textarea = screen.getByPlaceholderText(/请输入您的查询问题/);
      await user.type(textarea, '测试优化');
      
      const optimizeButton = screen.getByRole('button', { name: /优化查询/ });
      await user.click(optimizeButton);
      
      await waitFor(() => {
        const applyButton = screen.getByRole('button', { name: /应用优化/ });
        return user.click(applyButton);
      });
      
      expect(mockStoreState.setCurrentQuery).toHaveBeenCalledWith('测试优化 智能检索');
      expect(message.success).toHaveBeenCalledWith('已应用优化建议');
    });
  });

  describe('快速查询', () => {
    it('应该处理快速查询点击', async () => {
      const user = userEvent.setup();
      render(<AgenticQueryPanel />);
      
      const quickQueryButton = screen.getByText('如何实现多智能体协作?');
      await user.click(quickQueryButton);
      
      expect(mockStoreState.setCurrentQuery).toHaveBeenCalledWith('如何实现多智能体协作?');
    });
  });

  describe('高级设置', () => {
    it('应该切换高级设置面板', async () => {
      const user = userEvent.setup();
      render(<AgenticQueryPanel />);
      
      const settingButton = screen.getByRole('button', { name: /高级设置/ });
      await user.click(settingButton);
      
      expect(screen.getByText('高级配置')).toBeInTheDocument();
      expect(screen.getByText('查询扩展策略')).toBeInTheDocument();
    });

    it('应该配置扩展策略', async () => {
      const user = userEvent.setup();
      render(<AgenticQueryPanel />);
      
      // 打开高级设置
      const settingButton = screen.getByRole('button', { name: /高级设置/ });
      await user.click(settingButton);
      
      // 打开扩展策略面板
      const expansionPanel = screen.getByText('查询扩展策略');
      await user.click(expansionPanel);
      
      // 选择扩展策略
      const strategySelect = screen.getByPlaceholderText('选择查询扩展策略');
      await user.click(strategySelect);
      
      const synonymOption = screen.getByText('同义词扩展');
      await user.click(synonymOption);
      
      expect(strategySelect).toBeInTheDocument();
    });

    it('应该配置检索策略', async () => {
      const user = userEvent.setup();
      render(<AgenticQueryPanel />);
      
      // 打开高级设置
      const settingButton = screen.getByRole('button', { name: /高级设置/ });
      await user.click(settingButton);
      
      // 打开检索策略面板
      const retrievalPanel = screen.getByText('检索策略');
      await user.click(retrievalPanel);
      
      // 应该显示检索方法选择器
      expect(screen.getByText('检索方法:')).toBeInTheDocument();
      expect(screen.getByText('结果数量:')).toBeInTheDocument();
      expect(screen.getByText('分数阈值:')).toBeInTheDocument();
    });

    it('应该配置系统功能开关', async () => {
      const user = userEvent.setup();
      render(<AgenticQueryPanel />);
      
      // 打开高级设置
      const settingButton = screen.getByRole('button', { name: /高级设置/ });
      await user.click(settingButton);
      
      // 打开系统功能面板
      const featuresPanel = screen.getByText('系统功能');
      await user.click(featuresPanel);
      
      // 应该显示功能开关
      expect(screen.getByText('启用结果解释')).toBeInTheDocument();
      expect(screen.getByText('启用后备策略')).toBeInTheDocument();
    });
  });

  describe('加载状态', () => {
    it('应该在查询时显示加载状态', async () => {
      const user = userEvent.setup();
      const slowStore = {
        ...mockStoreState,
        isAgenticQuerying: true,
      };
      (useRagStore as unknown as Mock).mockReturnValue(slowStore);
      
      render(<AgenticQueryPanel />);
      
      const textarea = screen.getByPlaceholderText(/请输入您的查询问题/);
      await user.type(textarea, '测试查询');
      
      const searchButton = screen.getByRole('button', { name: /智能检索中/ });
      expect(searchButton).toBeDisabled();
    });

    it('应该在分析时显示进度指示器', async () => {
      vi.useFakeTimers();
      const user = userEvent.setup();
      render(<AgenticQueryPanel autoAnalyze />);
      
      const textarea = screen.getByPlaceholderText(/请输入您的查询问题/);
      await user.type(textarea, '代码实现');
      
      vi.advanceTimersByTime(1000);
      
      // 在分析过程中应该显示进度
      await waitFor(() => {
        const titleElement = screen.getByText('智能查询');
        expect(titleElement).toBeInTheDocument();
      });

      vi.useRealTimers();
    });
  });

  describe('错误处理', () => {
    it('应该处理查询分析错误', async () => {
      // 创建一个会抛出错误的分析函数
      vi.useFakeTimers();
      const user = userEvent.setup();
      render(<AgenticQueryPanel autoAnalyze />);
      
      // 模拟网络错误
      const originalPromise = window.Promise;
      window.Promise = class extends originalPromise {
        static resolve = vi.fn().mockRejectedValue(new Error('分析失败'));
      } as any;
      
      const textarea = screen.getByPlaceholderText(/请输入您的查询问题/);
      await user.type(textarea, '测试查询');
      
      vi.advanceTimersByTime(1000);
      
      // 恢复原始Promise
      window.Promise = originalPromise;
      vi.useRealTimers();
    });

    it('应该在查询优化失败时处理错误', async () => {
      const user = userEvent.setup();
      render(<AgenticQueryPanel />);
      
      // 模拟优化失败
      const textarea = screen.getByPlaceholderText(/请输入您的查询问题/);
      await user.type(textarea, '');  // 空查询不会触发优化
      
      const optimizeButton = screen.getByRole('button', { name: /优化查询/ });
      expect(optimizeButton).toBeDisabled();
    });
  });

  describe('回调函数', () => {
    it('应该正确调用onSearch回调', async () => {
      const user = userEvent.setup();
      const onSearch = vi.fn();
      
      render(<AgenticQueryPanel onSearch={onSearch} />);
      
      const textarea = screen.getByPlaceholderText(/请输入您的查询问题/);
      await user.type(textarea, '测试查询');
      
      const searchButton = screen.getByRole('button', { name: /智能检索/ });
      await user.click(searchButton);
      
      await waitFor(() => {
        expect(onSearch).toHaveBeenCalledWith(
          expect.objectContaining({
            query: '测试查询',
            context_history: ['测试历史'],
            retrieval_strategies: ['semantic', 'keyword'],
            max_results: 10,
            include_explanation: true,
          })
        );
      });
    });

    it('应该正确调用onResults回调', async () => {
      const user = userEvent.setup();
      const onResults = vi.fn();
      const mockResponse = {
        success: true,
        analysis_info: { intent: 'factual' },
        expanded_queries: ['测试扩展查询'],
      };
      (ragService.agenticQuery as Mock).mockResolvedValue(mockResponse);
      
      render(<AgenticQueryPanel onResults={onResults} />);
      
      const textarea = screen.getByPlaceholderText(/请输入您的查询问题/);
      await user.type(textarea, '测试查询');
      
      const searchButton = screen.getByRole('button', { name: /智能检索/ });
      await user.click(searchButton);
      
      await waitFor(() => {
        expect(onResults).toHaveBeenCalledWith(mockResponse);
      });
    });
  });

  describe('可访问性', () => {
    it('应该具有正确的ARIA标签', () => {
      render(<AgenticQueryPanel />);
      
      const textarea = screen.getByPlaceholderText(/请输入您的查询问题/);
      expect(textarea).toHaveAttribute('aria-label', undefined); // 默认没有aria-label

      const searchButton = screen.getByRole('button', { name: /智能检索/ });
      expect(searchButton).toBeInTheDocument();
    });

    it('应该支持键盘导航', async () => {
      const user = userEvent.setup();
      render(<AgenticQueryPanel />);
      
      const textarea = screen.getByPlaceholderText(/请输入您的查询问题/);
      await user.type(textarea, '测试查询');
      
      // Tab键导航到搜索按钮
      await user.tab();
      const searchButton = screen.getByRole('button', { name: /智能检索/ });
      expect(searchButton).toHaveFocus();
    });
  });
});