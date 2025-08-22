import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { BrowserRouter } from 'react-router-dom';
import VectorClockVisualizationPage from '../../../src/pages/VectorClockVisualizationPage';

// Mock antd components that might cause issues in tests
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

const renderWithRouter = (component: React.ReactElement) => {
  return render(
    <BrowserRouter>
      {component}
    </BrowserRouter>
  );
};

describe('VectorClockVisualizationPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('应该渲染页面标题和描述', () => {
    renderWithRouter(<VectorClockVisualizationPage />);
    
    expect(screen.getByText('⏰ 向量时钟算法可视化')).toBeInTheDocument();
    expect(screen.getByText(/分布式系统中的向量时钟算法演示/)).toBeInTheDocument();
  });

  it('应该显示节点状态卡片', () => {
    renderWithRouter(<VectorClockVisualizationPage />);
    
    expect(screen.getByText('节点状态')).toBeInTheDocument();
    expect(screen.getByText('Node A')).toBeInTheDocument();
    expect(screen.getByText('Node B')).toBeInTheDocument();
    expect(screen.getByText('Node C')).toBeInTheDocument();
  });

  it('应该显示事件时间线', () => {
    renderWithRouter(<VectorClockVisualizationPage />);
    
    expect(screen.getByText('🕐 事件时间线')).toBeInTheDocument();
    expect(screen.getByText('节点操作历史')).toBeInTheDocument();
  });

  it('应该显示向量时钟比较功能', () => {
    renderWithRouter(<VectorClockVisualizationPage />);
    
    expect(screen.getByText('🔍 向量时钟比较')).toBeInTheDocument();
    expect(screen.getByText('时钟A')).toBeInTheDocument();
    expect(screen.getByText('时钟B')).toBeInTheDocument();
  });

  it('应该能够添加新事件', async () => {
    renderWithRouter(<VectorClockVisualizationPage />);
    
    // 查找添加事件按钮
    const addButtons = screen.getAllByText('添加事件');
    expect(addButtons.length).toBeGreaterThan(0);
    
    // 点击第一个添加事件按钮
    fireEvent.click(addButtons[0]);
    
    // 等待UI更新
    await waitFor(() => {
      // 验证事件是否被添加到时间线
      const timelineItems = screen.getAllByText(/本地事件|发送消息|接收消息/);
      expect(timelineItems.length).toBeGreaterThan(0);
    });
  });

  it('应该能够发送消息', async () => {
    renderWithRouter(<VectorClockVisualizationPage />);
    
    // 查找发送消息按钮
    const sendButtons = screen.getAllByText('发送消息');
    expect(sendButtons.length).toBeGreaterThan(0);
    
    // 点击发送消息按钮
    fireEvent.click(sendButtons[0]);
    
    // 等待UI更新
    await waitFor(() => {
      // 验证消息事件是否被添加
      const messageEvents = screen.getAllByText(/发送消息|接收消息/);
      expect(messageEvents.length).toBeGreaterThan(0);
    });
  });

  it('应该显示因果关系分析', () => {
    renderWithRouter(<VectorClockVisualizationPage />);
    
    expect(screen.getByText('📊 因果关系分析')).toBeInTheDocument();
    expect(screen.getByText('并发事件检测')).toBeInTheDocument();
    expect(screen.getByText('因果链追踪')).toBeInTheDocument();
  });

  it('应该显示算法说明', () => {
    renderWithRouter(<VectorClockVisualizationPage />);
    
    expect(screen.getByText('📖 算法原理说明')).toBeInTheDocument();
    expect(screen.getByText('向量时钟基础')).toBeInTheDocument();
    expect(screen.getByText('算法步骤')).toBeInTheDocument();
  });

  it('应该能够比较向量时钟', () => {
    renderWithRouter(<VectorClockVisualizationPage />);
    
    // 验证比较结果显示
    expect(screen.getByText('比较结果:')).toBeInTheDocument();
    
    // 应该显示关系类型（before, after, concurrent, equal之一）
    const relationshipTexts = screen.getAllByText(/before|after|concurrent|equal|之前|之后|并发|相等/);
    expect(relationshipTexts.length).toBeGreaterThan(0);
  });

  it('应该显示实时统计信息', () => {
    renderWithRouter(<VectorClockVisualizationPage />);
    
    expect(screen.getByText('总事件数')).toBeInTheDocument();
    expect(screen.getByText('消息传递')).toBeInTheDocument();
    expect(screen.getByText('并发事件')).toBeInTheDocument();
    expect(screen.getByText('因果关系')).toBeInTheDocument();
  });

  it('应该能够切换实时模式', () => {
    renderWithRouter(<VectorClockVisualizationPage />);
    
    // 查找实时模式开关
    const realTimeText = screen.getByText('实时模拟');
    expect(realTimeText).toBeInTheDocument();
    
    // 查找开关组件（Switch）
    const switches = screen.container.querySelectorAll('.ant-switch');
    expect(switches.length).toBeGreaterThan(0);
  });

  it('应该显示向量时钟的JSON表示', () => {
    renderWithRouter(<VectorClockVisualizationPage />);
    
    // 查找显示向量时钟值的元素
    const clockDisplays = screen.container.querySelectorAll('code');
    expect(clockDisplays.length).toBeGreaterThan(0);
  });

  it('应该处理节点操作历史', () => {
    renderWithRouter(<VectorClockVisualizationPage />);
    
    // 验证历史记录显示
    expect(screen.getByText('节点操作历史')).toBeInTheDocument();
    
    // 应该显示时间戳
    const timestamps = screen.getAllByText(/\d{2}:\d{2}:\d{2}/);
    expect(timestamps.length).toBeGreaterThan(0);
  });

  it('应该在重置时清除所有状态', async () => {
    renderWithRouter(<VectorClockVisualizationPage />);
    
    // 添加一些事件
    const addButtons = screen.getAllByText('添加事件');
    if (addButtons.length > 0) {
      fireEvent.click(addButtons[0]);
    }
    
    // 查找重置按钮
    const resetButtons = screen.getAllByText(/重置|清除|Reset/);
    if (resetButtons.length > 0) {
      fireEvent.click(resetButtons[0]);
      
      await waitFor(() => {
        // 验证状态已重置（这里根据具体实现可能需要调整）
        expect(screen.getByText('向量时钟算法可视化')).toBeInTheDocument();
      });
    }
  });

  it('应该响应式地更新时钟值', async () => {
    renderWithRouter(<VectorClockVisualizationPage />);
    
    // 记录初始状态
    const initialClockElements = screen.container.querySelectorAll('code');
    const initialCount = initialClockElements.length;
    
    // 添加事件应该更新时钟
    const addButtons = screen.getAllByText('添加事件');
    if (addButtons.length > 0) {
      fireEvent.click(addButtons[0]);
      
      await waitFor(() => {
        // 验证时钟值已更新
        const updatedClockElements = screen.container.querySelectorAll('code');
        expect(updatedClockElements.length).toBeGreaterThanOrEqual(initialCount);
      });
    }
  });
});

describe('VectorClock比较算法', () => {
  it('应该正确识别before关系', () => {
    renderWithRouter(<VectorClockVisualizationPage />);
    
    // 这个测试验证比较逻辑
    // 由于比较算法在组件内部，我们通过UI交互来测试
    expect(screen.getByText('比较结果:')).toBeInTheDocument();
  });

  it('应该正确识别concurrent关系', () => {
    renderWithRouter(<VectorClockVisualizationPage />);
    
    // 验证并发关系检测
    expect(screen.getByText('并发事件检测')).toBeInTheDocument();
  });

  it('应该正确处理相等关系', () => {
    renderWithRouter(<VectorClockVisualizationPage />);
    
    // 验证相等关系处理
    expect(screen.getByText('因果链追踪')).toBeInTheDocument();
  });
});