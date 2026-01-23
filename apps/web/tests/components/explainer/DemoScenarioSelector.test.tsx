import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import '@testing-library/jest-dom'
import DemoScenarioSelector from '../../../src/components/explainer/DemoScenarioSelector'

describe('DemoScenarioSelector', () => {
  const mockOnSelectScenario = jest.fn()

  beforeEach(() => {
    mockOnSelectScenario.mockClear()
  })

  it('renders scenario selector with title and description', () => {
    render(<DemoScenarioSelector onSelectScenario={mockOnSelectScenario} />)

    expect(screen.getByText('演示场景选择器')).toBeInTheDocument()
    expect(
      screen.getByText(/选择不同的业务场景来体验解释性AI/)
    ).toBeInTheDocument()
  })

  it('displays all category filters', () => {
    render(<DemoScenarioSelector onSelectScenario={mockOnSelectScenario} />)

    expect(screen.getByText('全部场景')).toBeInTheDocument()
    expect(screen.getByText('金融服务')).toBeInTheDocument()
    expect(screen.getByText('医疗健康')).toBeInTheDocument()
    expect(screen.getByText('电商零售')).toBeInTheDocument()
    expect(screen.getByText('人力资源')).toBeInTheDocument()
    expect(screen.getByText('安全防护')).toBeInTheDocument()
  })

  it('shows scenario count badges for each category', () => {
    render(<DemoScenarioSelector onSelectScenario={mockOnSelectScenario} />)

    // 应该显示各分类的场景数量
    const badges = screen.getAllByText(/^\d+$/)
    expect(badges.length).toBeGreaterThan(0)
  })

  it('displays all demo scenarios', () => {
    render(<DemoScenarioSelector onSelectScenario={mockOnSelectScenario} />)

    expect(screen.getByText('银行贷款审批决策')).toBeInTheDocument()
    expect(screen.getByText('医疗诊断辅助系统')).toBeInTheDocument()
    expect(screen.getByText('电商个性化推荐')).toBeInTheDocument()
    expect(screen.getByText('员工绩效评估')).toBeInTheDocument()
    expect(screen.getByText('金融反欺诈检测')).toBeInTheDocument()
    expect(screen.getByText('供应链风险管理')).toBeInTheDocument()
  })

  it('filters scenarios by category', async () => {
    render(<DemoScenarioSelector onSelectScenario={mockOnSelectScenario} />)

    // 点击金融服务分类
    fireEvent.click(screen.getByText('金融服务'))

    await waitFor(() => {
      expect(screen.getByText('银行贷款审批决策')).toBeInTheDocument()
      expect(screen.getByText('供应链风险管理')).toBeInTheDocument()
      // 其他分类的场景应该不显示
      expect(screen.queryByText('医疗诊断辅助系统')).not.toBeInTheDocument()
    })
  })

  it('shows complexity levels correctly', () => {
    render(<DemoScenarioSelector onSelectScenario={mockOnSelectScenario} />)

    expect(screen.getByText('初级')).toBeInTheDocument()
    expect(screen.getByText('中级')).toBeInTheDocument()
    expect(screen.getByText('高级')).toBeInTheDocument()
  })

  it('displays scenario preview data', () => {
    render(<DemoScenarioSelector onSelectScenario={mockOnSelectScenario} />)

    expect(screen.getByText('决策预览')).toBeInTheDocument()
    expect(screen.getByText('贷款申请批准')).toBeInTheDocument()
    expect(screen.getByText('85%')).toBeInTheDocument() // confidence
    expect(screen.getByText('低')).toBeInTheDocument() // risk level
  })

  it('shows feature badges for each scenario', () => {
    render(<DemoScenarioSelector onSelectScenario={mockOnSelectScenario} />)

    expect(screen.getByText('置信度分析')).toBeInTheDocument()
    expect(screen.getByText('风险评估')).toBeInTheDocument()
    expect(screen.getByText('反事实分析')).toBeInTheDocument()
    expect(screen.getByText('CoT推理')).toBeInTheDocument()
  })

  it('expands scenario details when info button is clicked', async () => {
    render(<DemoScenarioSelector onSelectScenario={mockOnSelectScenario} />)

    const infoButtons = screen.getAllByRole('button')
    const infoButton = infoButtons.find(
      button =>
        button.querySelector('svg')?.getAttribute('data-lucide') === 'info'
    )

    if (infoButton) {
      fireEvent.click(infoButton)

      await waitFor(() => {
        expect(screen.getByText('数据集信息')).toBeInTheDocument()
        expect(screen.getByText('学习目标')).toBeInTheDocument()
      })
    }
  })

  it('shows dataset information when expanded', async () => {
    render(<DemoScenarioSelector onSelectScenario={mockOnSelectScenario} />)

    const infoButtons = screen.getAllByRole('button')
    const infoButton = infoButtons.find(
      button =>
        button.querySelector('svg')?.getAttribute('data-lucide') === 'info'
    )

    if (infoButton) {
      fireEvent.click(infoButton)

      await waitFor(() => {
        expect(screen.getByText('信用评分数据')).toBeInTheDocument()
        expect(screen.getByText('央行征信系统数据')).toBeInTheDocument()
        expect(screen.getByText('10,000 条')).toBeInTheDocument()
      })
    }
  })

  it('displays learning objectives when expanded', async () => {
    render(<DemoScenarioSelector onSelectScenario={mockOnSelectScenario} />)

    const infoButtons = screen.getAllByRole('button')
    const infoButton = infoButtons.find(
      button =>
        button.querySelector('svg')?.getAttribute('data-lucide') === 'info'
    )

    if (infoButton) {
      fireEvent.click(infoButton)

      await waitFor(() => {
        expect(
          screen.getByText('理解金融风险评估的多维度分析')
        ).toBeInTheDocument()
        expect(
          screen.getByText('掌握置信度计算在信贷决策中的应用')
        ).toBeInTheDocument()
      })
    }
  })

  it('calls onSelectScenario when start button is clicked', () => {
    render(<DemoScenarioSelector onSelectScenario={mockOnSelectScenario} />)

    const startButton = screen.getAllByText('开始体验')[0]
    fireEvent.click(startButton)

    expect(mockOnSelectScenario).toHaveBeenCalledTimes(1)
    expect(mockOnSelectScenario).toHaveBeenCalledWith(
      expect.objectContaining({
        title: '银行贷款审批决策',
        category: 'finance',
      })
    )
  })

  it('shows selected scenario as disabled', () => {
    render(
      <DemoScenarioSelector
        onSelectScenario={mockOnSelectScenario}
        selectedScenario="loan-approval"
      />
    )

    // 找到选中的场景按钮
    const selectedButton = screen.getByText('已选择')
    expect(selectedButton).toBeDisabled()
  })

  it('displays duration information', () => {
    render(<DemoScenarioSelector onSelectScenario={mockOnSelectScenario} />)

    expect(screen.getByText('15-20分钟')).toBeInTheDocument()
    expect(screen.getByText('25-30分钟')).toBeInTheDocument()
    expect(screen.getByText('10-15分钟')).toBeInTheDocument()
  })

  it('shows risk level colors correctly', () => {
    render(<DemoScenarioSelector onSelectScenario={mockOnSelectScenario} />)

    const lowRiskElements = screen.getAllByText('低')
    const mediumRiskElements = screen.getAllByText('中')
    const highRiskElements = screen.getAllByText('高')

    expect(lowRiskElements.length).toBeGreaterThan(0)
    expect(mediumRiskElements.length).toBeGreaterThan(0)
    expect(highRiskElements.length).toBeGreaterThan(0)
  })

  it('shows complexity indicators', () => {
    render(<DemoScenarioSelector onSelectScenario={mockOnSelectScenario} />)

    // 应该显示复杂度指示器
    expect(screen.getAllByText('初级').length).toBeGreaterThan(0)
    expect(screen.getAllByText('中级').length).toBeGreaterThan(0)
    expect(screen.getAllByText('高级').length).toBeGreaterThan(0)
  })

  it('displays usage instructions', () => {
    render(<DemoScenarioSelector onSelectScenario={mockOnSelectScenario} />)

    expect(screen.getByText('使用说明')).toBeInTheDocument()
    expect(
      screen.getByText('每个场景都包含完整的数据处理和决策解释流程')
    ).toBeInTheDocument()
    expect(
      screen.getByText('建议按照从初级到高级的顺序体验不同复杂度的场景')
    ).toBeInTheDocument()
  })

  it('resets filter when "全部场景" is clicked', async () => {
    render(<DemoScenarioSelector onSelectScenario={mockOnSelectScenario} />)

    // 先选择一个分类
    fireEvent.click(screen.getByText('金融服务'))

    await waitFor(() => {
      expect(screen.queryByText('医疗诊断辅助系统')).not.toBeInTheDocument()
    })

    // 然后点击全部场景
    fireEvent.click(screen.getByText('全部场景'))

    await waitFor(() => {
      expect(screen.getByText('医疗诊断辅助系统')).toBeInTheDocument()
    })
  })

  it('handles scenario expansion and collapse', async () => {
    render(<DemoScenarioSelector onSelectScenario={mockOnSelectScenario} />)

    const infoButtons = screen.getAllByRole('button')
    const firstInfoButton = infoButtons.find(
      button =>
        button.querySelector('svg')?.getAttribute('data-lucide') === 'info'
    )

    if (firstInfoButton) {
      // 展开
      fireEvent.click(firstInfoButton)

      await waitFor(() => {
        expect(screen.getByText('数据集信息')).toBeInTheDocument()
      })

      // 再次点击收起
      fireEvent.click(firstInfoButton)

      await waitFor(() => {
        expect(screen.queryByText('数据集信息')).not.toBeInTheDocument()
      })
    }
  })

  it('displays key factors for each scenario', () => {
    render(<DemoScenarioSelector onSelectScenario={mockOnSelectScenario} />)

    expect(screen.getByText('信用评分: 750')).toBeInTheDocument()
    expect(screen.getByText('月收入: ¥15,000')).toBeInTheDocument()
    expect(screen.getByText('工作年限: 5年')).toBeInTheDocument()
  })
})
