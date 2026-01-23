import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import TabularQLearningPage from '../../src/pages/qlearning/TabularQLearningPage'

describe('TabularQLearningPage', () => {
  it('应该正确渲染页面标题和描述', () => {
    render(<TabularQLearningPage />)

    expect(
      screen.getByText('表格Q-Learning (Tabular Q-Learning)')
    ).toBeInTheDocument()
    expect(screen.getByText(/经典的表格式Q-Learning算法/)).toBeInTheDocument()
  })

  it('应该显示算法参数配置', () => {
    render(<TabularQLearningPage />)

    expect(screen.getByText(/学习率.*Learning Rate/)).toBeInTheDocument()
    expect(screen.getByText(/折扣因子.*Gamma/)).toBeInTheDocument()
    expect(screen.getByText(/探索率.*Epsilon/)).toBeInTheDocument()

    // 检查参数说明
    expect(
      screen.getByText('控制新信息对已有知识的影响程度')
    ).toBeInTheDocument()
    expect(screen.getByText('未来奖励的重要性权重')).toBeInTheDocument()
    expect(screen.getByText('随机探索vs贪婪利用的平衡')).toBeInTheDocument()
  })

  it('应该显示Q-Learning核心原理', () => {
    render(<TabularQLearningPage />)

    expect(screen.getByText('Q-Learning更新公式')).toBeInTheDocument()
    expect(screen.getByText(/Q\(s,a\).*←.*Q\(s,a\)/)).toBeInTheDocument()

    // 检查算法标签
    expect(screen.getByText('Model-Free')).toBeInTheDocument()
    expect(screen.getByText('Off-Policy')).toBeInTheDocument()
    expect(screen.getByText('Value-Based')).toBeInTheDocument()
  })

  it('应该能够开始训练', async () => {
    render(<TabularQLearningPage />)

    const startBtn = screen.getByRole('button', { name: /开始训练/ })
    fireEvent.click(startBtn)

    // 验证训练开始
    await waitFor(() => {
      expect(screen.getByText('正在训练中...')).toBeInTheDocument()
      expect(startBtn).toBeDisabled()
    })
  })

  it('应该能够重置训练状态', () => {
    render(<TabularQLearningPage />)

    const resetBtn = screen.getByRole('button', { name: /重置/ })
    fireEvent.click(resetBtn)

    // 验证训练回合重置为0
    expect(screen.getByDisplayValue('0')).toBeInTheDocument()
  })

  it('应该显示Q表结构', () => {
    render(<TabularQLearningPage />)

    expect(screen.getByText('Q表可视化 (4x4 GridWorld)')).toBeInTheDocument()
    expect(screen.getByText('状态')).toBeInTheDocument()
    expect(screen.getByText('↑ (向上)')).toBeInTheDocument()
    expect(screen.getByText('→ (向右)')).toBeInTheDocument()
    expect(screen.getByText('↓ (向下)')).toBeInTheDocument()
    expect(screen.getByText('← (向左)')).toBeInTheDocument()
  })

  it('应该显示算法特点和适用场景', () => {
    render(<TabularQLearningPage />)

    // 检查优点
    expect(screen.getByText('原理简单，易于理解和实现')).toBeInTheDocument()
    expect(screen.getByText('理论基础扎实，有收敛性保证')).toBeInTheDocument()

    // 检查缺点
    expect(screen.getByText('只适用于小规模离散状态空间')).toBeInTheDocument()
    expect(screen.getByText('无法处理连续状态空间')).toBeInTheDocument()

    // 检查适用场景
    expect(screen.getByText('小规模网格世界 (GridWorld)')).toBeInTheDocument()
    expect(
      screen.getByText('简单的游戏环境 (如Tic-Tac-Toe)')
    ).toBeInTheDocument()
  })

  it('应该能够调整参数滑块', () => {
    render(<TabularQLearningPage />)

    // 测试学习率滑块调整
    const learningRateSlider = screen.getAllByRole('slider')[0]
    fireEvent.change(learningRateSlider, { target: { value: '0.5' } })

    // 测试gamma滑块调整
    const gammaSlider = screen.getAllByRole('slider')[1]
    fireEvent.change(gammaSlider, { target: { value: '0.8' } })

    // 测试epsilon滑块调整
    const epsilonSlider = screen.getAllByRole('slider')[2]
    fireEvent.change(epsilonSlider, { target: { value: '0.3' } })
  })
})
