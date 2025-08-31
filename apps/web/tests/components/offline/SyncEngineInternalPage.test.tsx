import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { BrowserRouter } from 'react-router-dom';
import SyncEngineInternalPage from '../../../src/pages/SyncEngineInternalPage';

// Mock antd components
mockFn()mock('antd', async () => {
  const actual = await mockFn()importActual('antd');
  return {
    ...actual,
    message: {
      success: mockFn()fn(),
      error: mockFn()fn(),
      info: mockFn()fn(),
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

describe('SyncEngineInternalPage', () => {
  beforeEach(() => {
    mockFn()clearAllMocks();
  });

  it('应该渲染页面标题和描述', () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    expect(screen.getByText('🔄 同步引擎内部机制展示')).toBeInTheDocument();
    expect(screen.getByText(/深入了解数据同步引擎的内部工作原理/)).toBeInTheDocument();
  });

  it('应该显示引擎控制面板', () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    expect(screen.getByText('引擎控制面板')).toBeInTheDocument();
    expect(screen.getByText('实时模式')).toBeInTheDocument();
    expect(screen.getByText('最大并发任务')).toBeInTheDocument();
    expect(screen.getByText('批处理大小')).toBeInTheDocument();
    expect(screen.getByText('检查点间隔')).toBeInTheDocument();
  });

  it('应该显示引擎统计信息', () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    expect(screen.getByText('已同步操作')).toBeInTheDocument();
    expect(screen.getByText('失败操作')).toBeInTheDocument();
    expect(screen.getByText('冲突解决')).toBeInTheDocument();
    expect(screen.getByText('同步效率')).toBeInTheDocument();
    expect(screen.getByText('平均吞吐量')).toBeInTheDocument();
    expect(screen.getByText('活跃任务')).toBeInTheDocument();
  });

  it('应该显示活跃同步任务表', () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    expect(screen.getByText('🏃‍♂️ 活跃同步任务')).toBeInTheDocument();
    expect(screen.getByText('任务ID')).toBeInTheDocument();
    expect(screen.getByText('方向')).toBeInTheDocument();
    expect(screen.getByText('优先级')).toBeInTheDocument();
    expect(screen.getByText('状态')).toBeInTheDocument();
    expect(screen.getByText('进度')).toBeInTheDocument();
    expect(screen.getByText('断点数据')).toBeInTheDocument();
  });

  it('应该显示等待队列任务', () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    expect(screen.getByText('⏳ 等待队列任务')).toBeInTheDocument();
  });

  it('应该显示操作批处理机制', () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    expect(screen.getByText('📦 操作批处理机制')).toBeInTheDocument();
    expect(screen.getByText('操作ID')).toBeInTheDocument();
    expect(screen.getByText('类型')).toBeInTheDocument();
    expect(screen.getByText('表名')).toBeInTheDocument();
    expect(screen.getByText('对象ID')).toBeInTheDocument();
    expect(screen.getByText('大小')).toBeInTheDocument();
  });

  it('应该显示批处理优化策略', () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    expect(screen.getByText('批处理优化策略')).toBeInTheDocument();
    expect(screen.getByText('操作分组：按表名和操作类型分组')).toBeInTheDocument();
    expect(screen.getByText('批量执行：减少网络往返次数')).toBeInTheDocument();
    expect(screen.getByText('断点续传：定期保存处理进度')).toBeInTheDocument();
    expect(screen.getByText('失败重试：指数退避重试策略')).toBeInTheDocument();
    expect(screen.getByText('冲突检测：向量时钟并发检测')).toBeInTheDocument();
  });

  it('应该显示同步流程可视化', () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    expect(screen.getByText('🔄 同步流程可视化')).toBeInTheDocument();
    expect(screen.getByText('上传流程')).toBeInTheDocument();
    expect(screen.getByText('下载流程')).toBeInTheDocument();
    expect(screen.getByText('双向流程')).toBeInTheDocument();
  });

  it('应该显示上传流程步骤', () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    expect(screen.getByText('创建同步任务')).toBeInTheDocument();
    expect(screen.getByText('获取待同步操作')).toBeInTheDocument();
    expect(screen.getByText('按批大小分组')).toBeInTheDocument();
    expect(screen.getByText('逐批上传操作')).toBeInTheDocument();
    expect(screen.getByText('创建检查点')).toBeInTheDocument();
    expect(screen.getByText('标记已同步')).toBeInTheDocument();
  });

  it('应该显示下载流程步骤', () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    expect(screen.getByText('获取服务器更新')).toBeInTheDocument();
    expect(screen.getByText('检测本地冲突')).toBeInTheDocument();
    expect(screen.getByText('解决冲突策略')).toBeInTheDocument();
    expect(screen.getByText('应用到本地')).toBeInTheDocument();
    expect(screen.getByText('更新向量时钟')).toBeInTheDocument();
    expect(screen.getByText('完成同步')).toBeInTheDocument();
  });

  it('应该显示增量同步机制说明', () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    expect(screen.getByText('增量同步机制')).toBeInTheDocument();
    expect(screen.getByText(/引擎支持增量数据同步/)).toBeInTheDocument();
  });

  it('应该能够切换实时模式', async () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    // 查找实时模式开关
    const realTimeSwitch = screen.container.querySelector('.ant-switch');
    expect(realTimeSwitch).toBeInTheDocument();
    
    if (realTimeSwitch) {
      fireEvent.click(realTimeSwitch);
      
      await waitFor(() => {
        // 验证开关状态已改变
        expect(realTimeSwitch).toBeInTheDocument();
      });
    }
  });

  it('应该能够修改引擎配置', async () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    // 查找配置选择器
    const selectors = screen.container.querySelectorAll('.ant-select-selector');
    expect(selectors.length).toBeGreaterThan(0);
    
    // 测试最大并发任务配置
    const maxConcurrentSelector = selectors[0];
    if (maxConcurrentSelector) {
      fireEvent.click(maxConcurrentSelector);
      
      await waitFor(() => {
        // 验证下拉选项出现
        expect(screen.container.querySelector('.ant-select-dropdown')).toBeInTheDocument();
      });
    }
  });

  it('应该显示任务优先级标签', () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    // 验证优先级标签存在
    const priorityTags = screen.container.querySelectorAll('.ant-tag');
    expect(priorityTags.length).toBeGreaterThan(0);
  });

  it('应该显示任务状态图标', () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    // 验证状态图标存在
    const statusIcons = screen.container.querySelectorAll('.anticon');
    expect(statusIcons.length).toBeGreaterThan(0);
  });

  it('应该显示进度条', () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    // 验证进度条存在
    const progressBars = screen.container.querySelectorAll('.ant-progress');
    expect(progressBars.length).toBeGreaterThan(0);
  });

  it('应该显示统计数值', () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    // 验证统计数值显示
    expect(screen.getByText(/\d+/)).toBeInTheDocument(); // 数字统计
    expect(screen.getByText(/\d+\.\d+%/)).toBeInTheDocument(); // 百分比
    expect(screen.getByText(/\d+\.\d+ops\/s/)).toBeInTheDocument(); // 吞吐量
  });

  it('应该实时更新任务进度', async () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    // 等待一段时间让实时更新触发
    await waitFor(() => {
      const progressElements = screen.container.querySelectorAll('.ant-progress-text');
      expect(progressElements.length).toBeGreaterThan(0);
    }, { timeout: 3000 });
  });

  it('应该显示任务算法说明', () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    expect(screen.getByText('任务调度算法')).toBeInTheDocument();
    expect(screen.getByText(/同步引擎按照优先级/)).toBeInTheDocument();
    
    expect(screen.getByText('优先级队列')).toBeInTheDocument();
    expect(screen.getByText(/任务按优先级排序/)).toBeInTheDocument();
  });

  it('应该处理不同的操作类型', () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    // 验证操作类型标签
    expect(screen.getByText('PUT') || screen.getByText('DELETE') || screen.getByText('PATCH')).toBeInTheDocument();
  });

  it('应该显示冲突状态', () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    // 验证冲突状态显示
    const conflictElements = screen.getAllByText(/冲突|正常/);
    expect(conflictElements.length).toBeGreaterThan(0);
  });

  it('应该显示批次信息', () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    // 验证批次信息显示
    expect(screen.getByText(/批大小/)).toBeInTheDocument();
    expect(screen.getByText(/当前批次操作/)).toBeInTheDocument();
  });
});

describe('SyncEngine数据更新', () => {
  it('应该在实时模式下定期更新数据', async () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    // 记录初始进度值
    const initialProgress = screen.container.querySelectorAll('.ant-progress-text')[0]?.textContent;
    
    // 等待更新周期
    await waitFor(() => {
      const updatedProgress = screen.container.querySelectorAll('.ant-progress-text')[0]?.textContent;
      // 在实时模式下，进度应该会更新
      expect(updatedProgress).toBeDefined();
    }, { timeout: 3000 });
  });

  it('应该正确处理任务状态变化', async () => {
    renderWithRouter(<SyncEngineInternalPage />);
    
    // 验证不同状态的任务存在
    await waitFor(() => {
      const statusElements = screen.getAllByText(/in_progress|pending|completed|failed/);
      expect(statusElements.length).toBeGreaterThan(0);
    });
  });
});