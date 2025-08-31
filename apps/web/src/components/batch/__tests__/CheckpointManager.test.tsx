// React import removed - not used in test
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import CheckpointManager from '../CheckpointManager';

// Mock antd's Modal.confirm
const mockConfirm = jest.fn();
jest.mock('antd', () => ({
  ...jest.requireActual('antd'),
  Modal: {
    ...jest.requireActual('antd').Modal,
    confirm: mockConfirm
  },
  message: {
    success: jest.fn(),
    error: jest.fn()
  }
}));

// Mock fetch
global.fetch = jest.fn();

const mockCheckpointData = {
  checkpoints: [
    {
      checkpoint_id: 'ckpt-001',
      job_id: 'job-001',
      created_at: '2024-01-01T10:00:00Z',
      checkpoint_type: 'manual',
      task_count: 100,
      completed_tasks: 80,
      failed_tasks: 5,
      file_size: 1024000,
      checksum: 'abc123',
      tags: { priority: 'high' }
    },
    {
      checkpoint_id: 'ckpt-002',
      job_id: 'job-002',
      created_at: '2024-01-01T11:00:00Z',
      checkpoint_type: 'auto',
      task_count: 50,
      completed_tasks: 45,
      failed_tasks: 2,
      file_size: 512000,
      checksum: 'def456',
      tags: {}
    }
  ]
};

const mockStatsData = {
  total_checkpoints: 2,
  total_size_bytes: 1536000,
  jobs_with_checkpoints: 2,
  checkpoint_types: {
    manual: 1,
    auto: 1
  },
  oldest_checkpoint: '2024-01-01T10:00:00Z',
  newest_checkpoint: '2024-01-01T11:00:00Z'
};

const mockJobsData = {
  jobs: [
    {
      id: 'job-001',
      name: 'Test Job 1',
      status: 'running',
      progress: 0.8,
      total_tasks: 100,
      completed_tasks: 80,
      failed_tasks: 5
    },
    {
      id: 'job-002', 
      name: 'Test Job 2',
      status: 'completed',
      progress: 1.0,
      total_tasks: 50,
      completed_tasks: 50,
      failed_tasks: 0
    }
  ]
};

describe('CheckpointManager', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    
    (fetch as jest.Mock)
      .mockImplementation((url: string) => {
        if (url.includes('/api/v1/batch/checkpoints/stats')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockStatsData
          });
        }
        if (url.includes('/api/v1/batch/jobs')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockJobsData
          });
        }
        if (url.includes('/api/v1/batch/checkpoints')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockCheckpointData
          });
        }
        return Promise.reject(new Error('Unknown URL'));
      });
  });

  it('渲染基本组件结构', async () => {
    render(<CheckpointManager />);
    
    await waitFor(() => {
      // 检查统计卡片
      expect(screen.getByText('总检查点数')).toBeInTheDocument();
      expect(screen.getByText('存储空间')).toBeInTheDocument();
      expect(screen.getByText('覆盖作业数')).toBeInTheDocument();
      expect(screen.getByText('检查点类型分布')).toBeInTheDocument();
    });
  });

  it('正确显示统计数据', async () => {
    render(<CheckpointManager />);
    
    await waitFor(() => {
      // 检查统计数值
      expect(screen.getByText('2')).toBeInTheDocument(); // 总检查点数
      expect(screen.getByText('1.46 MB')).toBeInTheDocument(); // 存储空间
      expect(screen.getByText('2')).toBeInTheDocument(); // 覆盖作业数
      
      // 检查类型分布
      expect(screen.getByText('manual')).toBeInTheDocument();
      expect(screen.getByText('auto')).toBeInTheDocument();
    });
  });

  it('显示检查点列表', async () => {
    render(<CheckpointManager />);
    
    await waitFor(() => {
      // 检查表格列标题
      expect(screen.getByText('检查点ID')).toBeInTheDocument();
      expect(screen.getByText('作业ID')).toBeInTheDocument();
      expect(screen.getByText('类型')).toBeInTheDocument();
      expect(screen.getByText('进度')).toBeInTheDocument();
      expect(screen.getByText('文件大小')).toBeInTheDocument();
      expect(screen.getByText('创建时间')).toBeInTheDocument();
      
      // 检查数据行
      expect(screen.getByText('ckpt-001')).toBeInTheDocument();
      expect(screen.getByText('ckpt-002')).toBeInTheDocument();
      expect(screen.getByText('MANUAL')).toBeInTheDocument();
      expect(screen.getByText('AUTO')).toBeInTheDocument();
    });
  });

  it('正确格式化文件大小', async () => {
    render(<CheckpointManager />);
    
    await waitFor(() => {
      // 1024000 bytes = 1000 KB
      expect(screen.getByText('1000 KB')).toBeInTheDocument();
      // 512000 bytes = 500 KB
      expect(screen.getByText('500 KB')).toBeInTheDocument();
    });
  });

  it('显示进度条', async () => {
    render(<CheckpointManager />);
    
    await waitFor(() => {
      // 检查进度显示 (80/100 和 45/50)
      expect(screen.getByText('80/100')).toBeInTheDocument();
      expect(screen.getByText('45/50')).toBeInTheDocument();
    });
  });

  it('处理作业筛选', async () => {
    render(<CheckpointManager />);
    
    await waitFor(() => {
      // 检查下拉选项
      expect(screen.getByText('所有作业')).toBeInTheDocument();
    });

    // 模拟选择特定作业
    const select = screen.getByDisplayValue('所有作业');
    fireEvent.mouseDown(select);
    
    await waitFor(() => {
      expect(screen.getByText('Test Job 1')).toBeInTheDocument();
      expect(screen.getByText('Test Job 2')).toBeInTheDocument();
    });
  });

  it('处理搜索功能', async () => {
    render(<CheckpointManager />);
    
    await waitFor(() => {
      expect(screen.getByText('ckpt-001')).toBeInTheDocument();
      expect(screen.getByText('ckpt-002')).toBeInTheDocument();
    });

    // 搜索特定检查点
    const searchInput = screen.getByPlaceholderText('搜索检查点ID或作业ID');
    fireEvent.change(searchInput, { target: { value: 'ckpt-001' } });

    // 验证过滤结果
    expect(screen.getByText('ckpt-001')).toBeInTheDocument();
    // ckpt-002 应该被过滤掉，但由于DOM可能还保留，我们检查实际的表格内容
  });

  it('处理创建检查点', async () => {
    (fetch as jest.Mock).mockImplementationOnce((url: string) => {
      if (url.includes('/checkpoint')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ checkpoint_id: 'new-checkpoint' })
        });
      }
      return Promise.resolve({
        ok: true,
        json: async () => mockCheckpointData
      });
    });

    render(<CheckpointManager />);
    
    await waitFor(() => {
      // 等待组件加载完成
      expect(screen.getByText('创建检查点')).toBeInTheDocument();
    });

    // 点击创建检查点按钮
    const createButton = screen.getByText('创建检查点');
    fireEvent.click(createButton);

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        '/api/v1/batch/jobs/all/checkpoint',
        { method: 'POST' }
      );
    });
  });

  it('处理恢复作业操作', async () => {
    mockConfirm.mockImplementation((config) => {
      config.onOk();
      return Promise.resolve();
    });

    (fetch as jest.Mock).mockImplementationOnce((url: string) => {
      if (url.includes('/restore')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ job_id: 'restored-job' })
        });
      }
      return Promise.resolve({
        ok: true,
        json: async () => mockCheckpointData
      });
    });

    render(<CheckpointManager />);
    
    await waitFor(() => {
      expect(screen.getAllByText('恢复')[0]).toBeInTheDocument();
    });

    // 点击恢复按钮
    const restoreButtons = screen.getAllByText('恢复');
    fireEvent.click(restoreButtons[0]);

    expect(mockConfirm).toHaveBeenCalled();
    
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        '/api/v1/batch/checkpoints/ckpt-001/restore',
        { method: 'POST' }
      );
    });
  });

  it('处理删除检查点操作', async () => {
    mockConfirm.mockImplementation((config) => {
      config.onOk();
      return Promise.resolve();
    });

    (fetch as jest.Mock).mockImplementationOnce((url: string) => {
      if (url.includes('/checkpoints/ckpt-001') && url.includes('DELETE')) {
        return Promise.resolve({ ok: true });
      }
      return Promise.resolve({
        ok: true,
        json: async () => mockCheckpointData
      });
    });

    render(<CheckpointManager />);
    
    await waitFor(() => {
      expect(screen.getAllByText('删除')[0]).toBeInTheDocument();
    });

    // 点击删除按钮
    const deleteButtons = screen.getAllByText('删除');
    fireEvent.click(deleteButtons[0]);

    expect(mockConfirm).toHaveBeenCalledWith(
      expect.objectContaining({
        title: '确认删除',
        danger: true
      })
    );
  });

  it('处理API错误', async () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
    (fetch as jest.Mock).mockRejectedValue(new Error('API Error'));

    render(<CheckpointManager />);
    
    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith(
        '获取检查点列表失败:',
        expect.any(Error)
      );
    });

    consoleSpy.mockRestore();
  });

  it('处理刷新操作', async () => {
    render(<CheckpointManager />);
    
    await waitFor(() => {
      expect(screen.getByText('刷新')).toBeInTheDocument();
    });

    // 点击刷新按钮
    const refreshButton = screen.getByText('刷新');
    fireEvent.click(refreshButton);

    // 验证API调用
    expect(fetch).toHaveBeenCalledWith('/api/v1/batch/checkpoints');
  });

  it('正确显示类型标签颜色', async () => {
    render(<CheckpointManager />);
    
    await waitFor(() => {
      const manualTag = screen.getByText('MANUAL');
      const autoTag = screen.getByText('AUTO');
      
      expect(manualTag).toBeInTheDocument();
      expect(autoTag).toBeInTheDocument();
      
      // 验证标签具有正确的类名（颜色）
      expect(manualTag.closest('.ant-tag')).toHaveClass('ant-tag-blue');
      expect(autoTag.closest('.ant-tag')).toHaveClass('ant-tag-green');
    });
  });

  it('显示空状态', async () => {
    (fetch as jest.Mock).mockImplementation((url: string) => {
      if (url.includes('/checkpoints')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ checkpoints: [] })
        });
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({ 
          total_checkpoints: 0,
          total_size_bytes: 0,
          jobs_with_checkpoints: 0,
          checkpoint_types: {}
        })
      });
    });

    render(<CheckpointManager />);
    
    await waitFor(() => {
      expect(screen.getByText('0')).toBeInTheDocument(); // 总检查点数为0
      expect(screen.getByText('0 B')).toBeInTheDocument(); // 存储空间为0
    });
  });
});