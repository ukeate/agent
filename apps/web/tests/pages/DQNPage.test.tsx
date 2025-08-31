import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import DQNPage from '../../src/pages/qlearning/DQNPage';

// Mock Ant Design Charts
mockFn()mock('@ant-design/charts', () => ({
  Line: ({ data, ...props }: any) => (
    <div data-testid="line-chart">
      <div>Chart with {data?.length || 0} data points</div>
      <div>{JSON.stringify(props)}</div>
    </div>
  ),
}));

describe('DQNPage', () => {
  it('应该正确渲染页面标题和描述', () => {
    render(<DQNPage />);
    
    expect(screen.getByText('Deep Q-Network (DQN)')).toBeInTheDocument();
    expect(screen.getByText(/使用深度神经网络逼近Q函数的强化学习算法/)).toBeInTheDocument();
  });

  it('应该显示网络架构配置', () => {
    render(<DQNPage />);
    
    expect(screen.getByText('网络架构配置')).toBeInTheDocument();
    expect(screen.getByText('网络类型:')).toBeInTheDocument();
    
    // 检查网络类型选择器
    expect(screen.getByDisplayValue('标准 DQN')).toBeInTheDocument();
  });

  it('应该显示网络结构详情', () => {
    render(<DQNPage />);
    
    expect(screen.getByText('当前网络结构')).toBeInTheDocument();
    expect(screen.getByText('输入层: 84 x 84 x 4 (状态)')).toBeInTheDocument();
    expect(screen.getByText('卷积层1: 32 filters, 8x8, stride=4')).toBeInTheDocument();
    expect(screen.getByText('输出层: 4 actions')).toBeInTheDocument();
    
    // 检查网络参数
    expect(screen.getByText('1,686,548')).toBeInTheDocument(); // 总参数数量
    expect(screen.getByText('Adam')).toBeInTheDocument(); // 优化器
    expect(screen.getByText('0.00025')).toBeInTheDocument(); // 学习率
  });

  it('应该能够切换网络类型', async () => {
    render(<DQNPage />);
    
    const networkSelect = screen.getByRole('combobox');
    fireEvent.mouseDown(networkSelect);
    
    await waitFor(() => {
      const doubleOption = screen.getByText('Double DQN');
      fireEvent.click(doubleOption);
    });
    
    expect(screen.getByDisplayValue('Double DQN')).toBeInTheDocument();
  });

  it('应该显示训练状态监控', () => {
    render(<DQNPage />);
    
    expect(screen.getByText('训练状态监控')).toBeInTheDocument();
    expect(screen.getByText(/训练回合.*\/ 500/)).toBeInTheDocument();
    expect(screen.getByText(/经验回放缓冲区.*\/ 10000/)).toBeInTheDocument();
    expect(screen.getByText('目标网络更新')).toBeInTheDocument();
  });

  it('应该显示DQN核心技术', () => {
    render(<DQNPage />);
    
    expect(screen.getByText('DQN核心技术')).toBeInTheDocument();
    expect(screen.getByText('经验回放 (Experience Replay)')).toBeInTheDocument();
    expect(screen.getByText('目标网络 (Target Network)')).toBeInTheDocument();
    expect(screen.getByText('卷积神经网络 (CNN)')).toBeInTheDocument();
    
    // 检查损失函数
    expect(screen.getByText('DQN损失函数')).toBeInTheDocument();
    expect(screen.getByText(/L = E\[\(r \+ γ max Q_target/)).toBeInTheDocument();
  });

  it('应该能够开始和暂停训练', async () => {
    render(<DQNPage />);
    
    const trainBtn = screen.getByRole('button', { name: /开始训练/ });
    fireEvent.click(trainBtn);
    
    await waitFor(() => {
      expect(screen.getByText('暂停训练')).toBeInTheDocument();
    });
    
    // 再次点击应该暂停
    fireEvent.click(trainBtn);
    expect(screen.getByText('开始训练')).toBeInTheDocument();
  });

  it('应该显示训练曲线可视化', () => {
    render(<DQNPage />);
    
    expect(screen.getByText('损失函数曲线')).toBeInTheDocument();
    expect(screen.getByText('奖励收敛曲线')).toBeInTheDocument();
    
    // 初始状态应该显示提示文字
    expect(screen.getByText('开始训练以查看损失曲线')).toBeInTheDocument();
    expect(screen.getByText('开始训练以查看奖励曲线')).toBeInTheDocument();
  });

  it('应该显示DQN算法原理详解', () => {
    render(<DQNPage />);
    
    expect(screen.getByText('DQN算法原理详解')).toBeInTheDocument();
    
    // 检查标签页
    expect(screen.getByText('经验回放机制')).toBeInTheDocument();
    expect(screen.getByText('目标网络稳定化')).toBeInTheDocument();
    expect(screen.getByText('网络架构设计')).toBeInTheDocument();
  });

  it('应该能够切换算法原理标签页', async () => {
    render(<DQNPage />);
    
    // 点击目标网络标签页
    const targetNetworkTab = screen.getByRole('tab', { name: '目标网络稳定化' });
    fireEvent.click(targetNetworkTab);
    
    await waitFor(() => {
      expect(screen.getByText('目标网络 (Target Network)')).toBeInTheDocument();
      expect(screen.getByText('DQN使用两个网络来稳定训练过程：')).toBeInTheDocument();
    });
  });

  it('应该显示算法对比表格', () => {
    render(<DQNPage />);
    
    expect(screen.getByText('DQN vs 表格Q-Learning')).toBeInTheDocument();
    expect(screen.getByText('状态表示')).toBeInTheDocument();
    expect(screen.getByText('函数逼近')).toBeInTheDocument();
    expect(screen.getByText('泛化能力')).toBeInTheDocument();
    expect(screen.getByText('状态空间规模')).toBeInTheDocument();
  });

  it('应该显示DQN的优势与局限', () => {
    render(<DQNPage />);
    
    expect(screen.getByText('DQN的优势与局限')).toBeInTheDocument();
    
    // 检查优势
    expect(screen.getByText('能处理高维状态空间 (如图像)')).toBeInTheDocument();
    expect(screen.getByText('具有良好的泛化能力')).toBeInTheDocument();
    
    // 检查局限
    expect(screen.getByText('训练不稳定，容易发散')).toBeInTheDocument();
    expect(screen.getByText('过高估计Q值 (overestimation bias)')).toBeInTheDocument();
  });

  it('应该能够重置训练状态', () => {
    render(<DQNPage />);
    
    const resetBtn = screen.getByRole('button', { name: /重置/ });
    fireEvent.click(resetBtn);
    
    // 验证状态被重置
    expect(screen.getByText(/训练回合.*0 \/ 500/)).toBeInTheDocument();
  });
});