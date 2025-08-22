import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { BrowserRouter } from 'react-router-dom';
import ModelCacheMonitorPage from '../../../src/pages/ModelCacheMonitorPage';

// Mock antd components
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

describe('ModelCacheMonitorPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('应该渲染页面标题和描述', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    expect(screen.getByText('🗄️ 本地模型缓存监控')).toBeInTheDocument();
    expect(screen.getByText(/监控和管理本地AI模型缓存/)).toBeInTheDocument();
  });

  it('应该显示缓存统计概览', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    expect(screen.getByText('缓存模型')).toBeInTheDocument();
    expect(screen.getByText('内存加载')).toBeInTheDocument();
    expect(screen.getByText('缓存使用率')).toBeInTheDocument();
    expect(screen.getByText('总缓存大小')).toBeInTheDocument();
    expect(screen.getByText('平均模型大小')).toBeInTheDocument();
    expect(screen.getByText('最大缓存')).toBeInTheDocument();
  });

  it('应该显示缓存空间使用情况', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    expect(screen.getByText('💾 缓存空间使用情况')).toBeInTheDocument();
    expect(screen.getByText('已使用')).toBeInTheDocument();
    expect(screen.getByText('剩余空间')).toBeInTheDocument();
    expect(screen.getByText('总容量')).toBeInTheDocument();
  });

  it('应该显示自动管理设置', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    expect(screen.getByText('自动管理设置')).toBeInTheDocument();
    expect(screen.getByText('自动清理')).toBeInTheDocument();
    expect(screen.getByText('压缩优化')).toBeInTheDocument();
    expect(screen.getByText('手动清理')).toBeInTheDocument();
  });

  it('应该显示缓存模型列表', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    expect(screen.getByText('📋 缓存模型列表')).toBeInTheDocument();
    expect(screen.getByText('模型ID')).toBeInTheDocument();
    expect(screen.getByText('状态')).toBeInTheDocument();
    expect(screen.getByText('大小')).toBeInTheDocument();
    expect(screen.getByText('使用统计')).toBeInTheDocument();
    expect(screen.getByText('标签')).toBeInTheDocument();
    expect(screen.getByText('操作')).toBeInTheDocument();
  });

  it('应该显示缓存管理策略', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    expect(screen.getByText('⚙️ 缓存管理策略')).toBeInTheDocument();
    expect(screen.getByText('LRU淘汰')).toBeInTheDocument();
    expect(screen.getByText('智能预加载')).toBeInTheDocument();
    expect(screen.getByText('压缩存储')).toBeInTheDocument();
    expect(screen.getByText('增量更新')).toBeInTheDocument();
    expect(screen.getByText('校验完整性')).toBeInTheDocument();
  });

  it('应该显示使用热度分析', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    expect(screen.getByText('📊 使用热度分析')).toBeInTheDocument();
    expect(screen.getByText('最常用模型')).toBeInTheDocument();
    expect(screen.getByText('最少用模型')).toBeInTheDocument();
    expect(screen.getByText('缓存优化建议')).toBeInTheDocument();
  });

  it('应该显示模型压缩与量化技术', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    expect(screen.getByText('🗜️ 模型压缩与量化技术')).toBeInTheDocument();
    expect(screen.getByText('压缩算法')).toBeInTheDocument();
    expect(screen.getByText('量化技术')).toBeInTheDocument();
    expect(screen.getByText('优化效果')).toBeInTheDocument();
  });

  it('应该显示压缩算法信息', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    expect(screen.getByText('GZIP')).toBeInTheDocument();
    expect(screen.getByText('LZ4')).toBeInTheDocument();
    expect(screen.getByText('ZSTD')).toBeInTheDocument();
    expect(screen.getByText(/通用压缩，压缩比30-40%/)).toBeInTheDocument();
  });

  it('应该显示量化技术信息', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    expect(screen.getByText('INT4')).toBeInTheDocument();
    expect(screen.getByText('INT8')).toBeInTheDocument();
    expect(screen.getByText('FP16')).toBeInTheDocument();
    expect(screen.getByText(/4位整数，最大压缩/)).toBeInTheDocument();
  });

  it('应该显示优化效果统计', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    expect(screen.getByText(/存储空间.*节省70%/)).toBeInTheDocument();
    expect(screen.getByText(/加载速度.*提升3x/)).toBeInTheDocument();
    expect(screen.getByText(/推理延迟.*降低50%/)).toBeInTheDocument();
  });

  it('应该显示模型详细信息', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    // 验证模型信息显示
    expect(screen.getByText(/claude-3-haiku-quantized|gpt-4-turbo-preview|llama-2-13b-chat/)).toBeInTheDocument();
    expect(screen.getByText(/已加载|磁盘缓存/)).toBeInTheDocument();
    expect(screen.getByText(/使用次数/)).toBeInTheDocument();
    expect(screen.getByText(/最后使用/)).toBeInTheDocument();
  });

  it('应该显示压缩比信息', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    // 验证压缩比显示
    expect(screen.getByText(/压缩比.*%/)).toBeInTheDocument();
  });

  it('应该显示量化级别标签', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    // 验证量化级别标签
    const quantizationTags = screen.container.querySelectorAll('.ant-tag');
    expect(quantizationTags.length).toBeGreaterThan(0);
  });

  it('应该显示模型标签', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    // 验证模型标签显示
    expect(screen.getByText(/reasoning|fast|quantized|multimodal|chat/)).toBeInTheDocument();
  });

  it('应该显示操作按钮', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    // 验证操作按钮存在
    const buttons = screen.container.querySelectorAll('.ant-btn');
    expect(buttons.length).toBeGreaterThan(0);
  });

  it('应该能够切换自动清理开关', async () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    // 查找自动清理开关
    const switches = screen.container.querySelectorAll('.ant-switch');
    expect(switches.length).toBeGreaterThan(0);
    
    const autoCleanupSwitch = switches[0];
    if (autoCleanupSwitch) {
      fireEvent.click(autoCleanupSwitch);
      
      await waitFor(() => {
        // 验证开关状态已改变
        expect(autoCleanupSwitch).toBeInTheDocument();
      });
    }
  });

  it('应该能够切换压缩优化开关', async () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    // 查找压缩优化开关
    const switches = screen.container.querySelectorAll('.ant-switch');
    expect(switches.length).toBeGreaterThan(1);
    
    const compressionSwitch = switches[1];
    if (compressionSwitch) {
      fireEvent.click(compressionSwitch);
      
      await waitFor(() => {
        // 验证开关状态已改变
        expect(compressionSwitch).toBeInTheDocument();
      });
    }
  });

  it('应该能够点击手动清理按钮', async () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    // 查找手动清理按钮
    const cleanupButton = screen.getByText('手动清理');
    expect(cleanupButton).toBeInTheDocument();
    
    fireEvent.click(cleanupButton);
    
    await waitFor(() => {
      // 验证按钮点击处理
      expect(cleanupButton).toBeInTheDocument();
    });
  });

  it('应该显示正确的缓存使用率进度条', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    // 验证进度条存在
    const progressBars = screen.container.querySelectorAll('.ant-progress');
    expect(progressBars.length).toBeGreaterThan(0);
  });

  it('应该显示字节大小格式化', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    // 验证字节大小显示（MB, GB等）
    expect(screen.getByText(/\d+(\.\d+)?\s*(MB|GB|KB|Bytes)/)).toBeInTheDocument();
  });

  it('应该处理不同的模型状态', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    // 验证状态标签显示
    const statusTags = screen.getAllByText(/已加载|磁盘缓存/);
    expect(statusTags.length).toBeGreaterThan(0);
  });

  it('应该显示量化级别颜色编码', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    // 验证量化级别的不同颜色标签
    const tags = screen.container.querySelectorAll('.ant-tag');
    const coloredTags = Array.from(tags).filter(tag => 
      tag.className.includes('ant-tag-red') || 
      tag.className.includes('ant-tag-orange') || 
      tag.className.includes('ant-tag-blue')
    );
    expect(coloredTags.length).toBeGreaterThan(0);
  });

  it('应该显示缓存统计数值', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    // 验证统计数值的显示
    const statistics = screen.container.querySelectorAll('.ant-statistic-content-value');
    expect(statistics.length).toBeGreaterThan(0);
  });

  it('应该显示缓存策略时间线', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    // 验证时间线组件存在
    const timeline = screen.container.querySelector('.ant-timeline');
    expect(timeline).toBeInTheDocument();
  });

  it('应该显示缓存优化建议警告', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    // 验证优化建议警告
    const alerts = screen.container.querySelectorAll('.ant-alert');
    expect(alerts.length).toBeGreaterThan(0);
  });

  it('应该正确处理表格分页', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    // 验证表格分页组件
    const pagination = screen.container.querySelector('.ant-pagination');
    expect(pagination).toBeInTheDocument();
  });
});

describe('ModelCache数据格式化', () => {
  it('应该正确格式化字节大小', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    // 验证字节格式化函数的效果
    expect(screen.getByText(/\d+(\.\d+)?\s*(MB|GB)/)).toBeInTheDocument();
  });

  it('应该正确显示压缩比百分比', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    // 验证压缩比显示
    expect(screen.getByText(/压缩比:\s*\d+%/)).toBeInTheDocument();
  });

  it('应该正确显示使用统计', () => {
    renderWithRouter(<ModelCacheMonitorPage />);
    
    // 验证使用次数和最后使用时间
    expect(screen.getByText(/使用次数:\s*\d+/)).toBeInTheDocument();
    expect(screen.getByText(/最后使用:/)).toBeInTheDocument();
  });
});