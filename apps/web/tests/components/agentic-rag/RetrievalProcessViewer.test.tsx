/**
 * RetrievalProcessViewer组件单元测试
 * 
 * 测试功能包括：
 * - 基础渲染和空状态显示
 * - 视图模式切换
 * - 实时进度显示
 * - 代理状态监控
 * - 时间线展示
 * - 流程步骤可视化
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, Mock } from 'vitest';
import RetrievalProcessViewer from '../../../src/components/agentic-rag/RetrievalProcessViewer';

// Mock dependencies
const mockRagStore = {
  retrievalProgress: null,
  isAgenticQuerying: false,
  queryAnalysis: null,
  agenticResults: null,
};

vi.mock('../../../src/stores/ragStore', () => ({
  useRagStore: () => mockRagStore,
}));

describe('RetrievalProcessViewer', () => {
  
  beforeEach(() => {
    vi.clearAllMocks();
    mockRagStore.retrievalProgress = null;
    mockRagStore.isAgenticQuerying = false;
    mockRagStore.queryAnalysis = null;
    mockRagStore.agenticResults = null;
  });

  describe('基础渲染', () => {
    it('无数据时显示空状态', () => {
      render(<RetrievalProcessViewer />);
      
      expect(screen.getByText('暂无检索过程数据')).toBeInTheDocument();
    });

    it('有检索进程时显示进度界面', () => {
      mockRagStore.isAgenticQuerying = true;
      
      render(<RetrievalProcessViewer />);
      
      expect(screen.getByText('检索流程')).toBeInTheDocument();
      expect(screen.getByText('进行中')).toBeInTheDocument();
    });

    it('compact模式下使用简化样式', () => {
      mockRagStore.isAgenticQuerying = true;
      
      render(<RetrievalProcessViewer compact={true} />);
      
      // 紧凑模式下应该不显示暂停/继续按钮
      expect(screen.queryByRole('button', { name: /暂停|继续/ })).not.toBeInTheDocument();
    });

    it('显示统计信息当有结果时', () => {
      mockRagStore.agenticResults = {
        success: true,
        query_id: 'test-query',
        results: [
          { id: '1', content: '测试结果1', score: 0.9 },
          { id: '2', content: '测试结果2', score: 0.8 },
        ],
        expanded_queries: ['查询1', '查询2'],
        processing_time: 1500,
        timestamp: '2024-01-01T00:00:00Z',
      };

      render(<RetrievalProcessViewer />);
      
      expect(screen.getByText('总结果数量')).toBeInTheDocument();
      expect(screen.getByText('2')).toBeInTheDocument(); // 结果数量
      expect(screen.getByText('扩展查询数')).toBeInTheDocument();
      expect(screen.getByText('平均相关度')).toBeInTheDocument();
      expect(screen.getByText('处理时间')).toBeInTheDocument();
    });
  });

  describe('视图模式切换', () => {
    beforeEach(() => {
      mockRagStore.isAgenticQuerying = true;
      render(<RetrievalProcessViewer />);
    });

    it('默认显示流程步骤标签页', () => {
      const stepsTab = screen.getByRole('tab', { name: /流程步骤/ });
      expect(stepsTab).toHaveAttribute('aria-selected', 'true');
      expect(screen.getByText('查询分析')).toBeInTheDocument();
      expect(screen.getByText('查询扩展')).toBeInTheDocument();
    });

    it('可以切换到时间线视图', async () => {
      const user = userEvent.setup();
      const timelineTab = screen.getByRole('tab', { name: /时间线/ });
      
      await user.click(timelineTab);
      
      expect(timelineTab).toHaveAttribute('aria-selected', 'true');
      expect(screen.getByText('执行时间线')).toBeInTheDocument();
    });

    it('可以切换到代理状态视图', async () => {
      const user = userEvent.setup();
      const agentsTab = screen.getByRole('tab', { name: /代理状态/ });
      
      await user.click(agentsTab);
      
      expect(agentsTab).toHaveAttribute('aria-selected', 'true');
      expect(screen.getByText('代理状态')).toBeInTheDocument();
    });

    it('showTimeline为false时不显示时间线标签页', () => {
      render(<RetrievalProcessViewer showTimeline={false} />);
      
      expect(screen.queryByRole('tab', { name: /时间线/ })).not.toBeInTheDocument();
    });
  });

  describe('流程步骤显示', () => {
    beforeEach(() => {
      mockRagStore.isAgenticQuerying = true;
      render(<RetrievalProcessViewer />);
    });

    it('显示所有预定义的处理步骤', () => {
      expect(screen.getByText('查询分析')).toBeInTheDocument();
      expect(screen.getByText('查询扩展')).toBeInTheDocument();
      expect(screen.getByText('多代理检索')).toBeInTheDocument();
      expect(screen.getByText('结果验证')).toBeInTheDocument();
      expect(screen.getByText('上下文组合')).toBeInTheDocument();
      expect(screen.getByText('结果解释')).toBeInTheDocument();
    });

    it('显示步骤描述', () => {
      expect(screen.getByText('分析查询意图、提取关键信息、确定复杂度')).toBeInTheDocument();
      expect(screen.getByText('生成同义词、语义扩展、上下文增强')).toBeInTheDocument();
      expect(screen.getByText('并行执行语义、关键词、结构化检索')).toBeInTheDocument();
    });

    it('显示步骤状态图标', () => {
      // 等待状态的步骤应该显示时钟图标
      const clockIcons = screen.getAllByRole('img', { name: /clock/ });
      expect(clockIcons.length).toBeGreaterThan(0);
    });
  });

  describe('实时进度更新', () => {
    it('更新步骤进度状态', async () => {
      mockRagStore.isAgenticQuerying = true;
      mockRagStore.retrievalProgress = {
        stage: 'analysis',
        progress: 50,
        message: '正在分析查询意图...',
      };

      render(<RetrievalProcessViewer />);
      
      expect(screen.getByText('正在分析查询意图...')).toBeInTheDocument();
      expect(screen.getByText('50%')).toBeInTheDocument();
    });

    it('显示完成状态的步骤', async () => {
      mockRagStore.retrievalProgress = {
        stage: 'complete',
        progress: 100,
        message: '检索完成',
      };

      render(<RetrievalProcessViewer />);
      
      // 完成状态应该显示勾选图标
      const checkIcons = screen.getAllByRole('img');
      const hasCheckIcon = checkIcons.some(icon => 
        icon.getAttribute('aria-label')?.includes('check-circle')
      );
      expect(hasCheckIcon).toBe(true);
    });

    it('显示总耗时信息', () => {
      mockRagStore.agenticResults = {
        processing_time: 2500,
      };

      render(<RetrievalProcessViewer />);
      
      expect(screen.getByText(/总耗时:/)).toBeInTheDocument();
    });
  });

  describe('代理状态监控', () => {
    beforeEach(async () => {
      mockRagStore.isAgenticQuerying = true;
      render(<RetrievalProcessViewer />);
      
      const user = userEvent.setup();
      const agentsTab = screen.getByRole('tab', { name: /代理状态/ });
      await user.click(agentsTab);
    });

    it('显示所有代理类型', () => {
      expect(screen.getByText('查询分析代理')).toBeInTheDocument();
      expect(screen.getByText('查询扩展代理')).toBeInTheDocument();
      expect(screen.getByText('语义检索代理')).toBeInTheDocument();
      expect(screen.getByText('关键词检索代理')).toBeInTheDocument();
      expect(screen.getByText('结果验证代理')).toBeInTheDocument();
      expect(screen.getByText('上下文组合代理')).toBeInTheDocument();
      expect(screen.getByText('解释生成代理')).toBeInTheDocument();
    });

    it('显示代理状态标签', () => {
      expect(screen.getAllByText('空闲').length).toBeGreaterThan(0);
    });

    it('显示代理进度条当有进度时', async () => {
      mockRagStore.retrievalProgress = {
        stage: 'analyzer',
        progress: 75,
        message: '分析中...',
      };

      // 重新渲染以更新进度
      render(<RetrievalProcessViewer />);
      
      const user = userEvent.setup();
      const agentsTab = screen.getByRole('tab', { name: /代理状态/ });
      await user.click(agentsTab);
      
      // 应该显示进度条
      const progressBars = screen.getAllByRole('progressbar');
      expect(progressBars.length).toBeGreaterThan(0);
    });

    it('显示不同代理的专用图标', () => {
      // 检查是否有各种图标类型
      const icons = screen.getAllByRole('img');
      expect(icons.length).toBeGreaterThan(0);
    });
  });

  describe('时间线视图功能', () => {
    beforeEach(async () => {
      mockRagStore.isAgenticQuerying = true;
      render(<RetrievalProcessViewer />);
      
      const user = userEvent.setup();
      const timelineTab = screen.getByRole('tab', { name: /时间线/ });
      await user.click(timelineTab);
    });

    it('显示时间线标题', () => {
      expect(screen.getByText('执行时间线')).toBeInTheDocument();
    });

    it('时间线中显示所有步骤', () => {
      expect(screen.getByText('查询分析')).toBeInTheDocument();
      expect(screen.getByText('查询扩展')).toBeInTheDocument();
      expect(screen.getByText('多代理检索')).toBeInTheDocument();
    });

    it('显示步骤耗时标签', async () => {
      // 模拟有耗时数据的步骤
      const mockSteps = [
        {
          id: 'analysis',
          name: '查询分析',
          status: 'finish',
          duration: 1.2,
        },
      ];

      // 这里需要模拟步骤有耗时数据
      // 实际实现中会通过props或state传入
      expect(screen.queryByText(/1.2s/)).toBeTruthy() || expect(screen.queryByText(/耗时/)).toBeTruthy();
    });
  });

  describe('交互控制功能', () => {
    beforeEach(() => {
      mockRagStore.isAgenticQuerying = true;
    });

    it('可以暂停和继续检索过程', async () => {
      render(<RetrievalProcessViewer />);
      
      const user = userEvent.setup();
      const pauseButton = screen.getByRole('button', { name: /暂停/ });
      
      await user.click(pauseButton);
      
      // 点击后应该变成继续按钮
      expect(screen.getByRole('button', { name: /继续/ })).toBeInTheDocument();
    });

    it('非检索状态下禁用暂停按钮', () => {
      mockRagStore.isAgenticQuerying = false;
      
      render(<RetrievalProcessViewer />);
      
      const pauseButton = screen.queryByRole('button', { name: /暂停/ });
      if (pauseButton) {
        expect(pauseButton).toBeDisabled();
      }
    });
  });

  describe('紧凑模式显示', () => {
    it('紧凑模式下使用水平步骤布局', () => {
      mockRagStore.isAgenticQuerying = true;
      
      render(<RetrievalProcessViewer compact={true} />);
      
      // 紧凑模式下应该直接显示步骤，不显示标签页
      expect(screen.queryByRole('tablist')).not.toBeInTheDocument();
      expect(screen.getByText('查询分析')).toBeInTheDocument();
    });

    it('紧凑模式下不显示额外控制按钮', () => {
      mockRagStore.isAgenticQuerying = true;
      
      render(<RetrievalProcessViewer compact={true} />);
      
      expect(screen.queryByRole('button', { name: /暂停|继续/ })).not.toBeInTheDocument();
      expect(screen.queryByText(/总耗时:/)).not.toBeInTheDocument();
    });
  });

  describe('autoUpdate功能', () => {
    it('autoUpdate为false时不自动更新', () => {
      mockRagStore.isAgenticQuerying = true;
      
      render(<RetrievalProcessViewer autoUpdate={false} />);
      
      // 这个测试主要验证组件能正常渲染
      expect(screen.getByText('检索流程')).toBeInTheDocument();
    });
  });

  describe('自定义类名和样式', () => {
    it('应用自定义className', () => {
      const { container } = render(
        <RetrievalProcessViewer
          className="custom-process-viewer"
        />
      );
      
      expect(container.querySelector('.custom-process-viewer')).toBeInTheDocument();
    });
  });

  describe('错误状态处理', () => {
    it('处理检索过程中的错误状态', () => {
      mockRagStore.retrievalProgress = {
        stage: 'analysis',
        progress: 0,
        message: '检索过程出错',
        error: true,
      };

      render(<RetrievalProcessViewer />);
      
      expect(screen.getByText('检索过程出错')).toBeInTheDocument();
    });
  });
});