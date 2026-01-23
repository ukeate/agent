import React from 'react'
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest'
import ModelEvaluationBenchmarkPage from '../../src/pages/ModelEvaluationBenchmarkPage'

// Mock Ant Design components
vi.mock('antd', async () => {
  const actual = await vi.importActual('antd')
  return {
    ...actual,
    message: {
      success: vi.fn(),
      error: vi.fn(),
    },
  }
})

// Mock API calls
const mockApiResponse = {
  benchmarks: [
    {
      id: 'glue-cola',
      name: 'cola',
      display_name: 'CoLA',
      description: 'Corpus of Linguistic Acceptability - 语言可接受性语料库',
      category: 'nlp',
      difficulty: 'medium',
      tasks: ['cola'],
      languages: ['en'],
      metrics: ['accuracy', 'f1'],
      num_samples: 1043,
      estimated_runtime_minutes: 5,
      memory_requirements_gb: 2.0,
      requires_gpu: false,
    },
    {
      id: 'mmlu',
      name: 'mmlu',
      display_name: 'MMLU',
      description:
        'Massive Multitask Language Understanding - 大规模多任务语言理解',
      category: 'knowledge',
      difficulty: 'hard',
      tasks: ['mmlu'],
      languages: ['en'],
      metrics: ['accuracy'],
      num_samples: 14042,
      estimated_runtime_minutes: 60,
      memory_requirements_gb: 8.0,
      requires_gpu: true,
    },
  ],
  models: [
    {
      id: 'bert-base-uncased',
      name: 'BERT Base Uncased',
      version: '1.0.0',
      description: 'BERT预训练模型，适用于各种NLP任务',
      model_path: '/models/bert-base-uncased',
      model_type: 'text_classification',
      architecture: 'transformer',
      parameters_count: 110000000,
    },
    {
      id: 'gpt-3.5-turbo',
      name: 'GPT-3.5 Turbo',
      version: '2023-12',
      description: 'OpenAI的GPT-3.5 Turbo模型',
      model_path: '/models/gpt-3.5-turbo',
      model_type: 'text_generation',
      architecture: 'transformer',
      parameters_count: 175000000000,
    },
  ],
  evaluationJobs: [
    {
      id: 'job_001',
      name: 'BERT在GLUE基准上的评估',
      status: 'running',
      progress: 0.65,
      current_task: '正在评估CoLA任务',
      models: [{ name: 'BERT Base Uncased' }],
      benchmarks: [{ name: 'CoLA' }, { name: 'SST-2' }],
      started_at: '2024-01-15T14:30:00Z',
      results: [],
      created_at: '2024-01-15T14:25:00Z',
    },
    {
      id: 'job_002',
      name: 'GPT模型代码生成评估',
      status: 'completed',
      progress: 1.0,
      models: [{ name: 'GPT-3.5 Turbo' }],
      benchmarks: [{ name: 'HumanEval' }],
      started_at: '2024-01-15T12:00:00Z',
      completed_at: '2024-01-15T12:45:00Z',
      results: [{ benchmark: 'HumanEval', accuracy: 0.734, pass_at_1: 0.456 }],
      created_at: '2024-01-15T11:55:00Z',
    },
  ],
}

describe('ModelEvaluationBenchmarkPage', () => {
  const user = userEvent.setup()

  beforeEach(() => {
    // Mock console methods to avoid noise in tests
    vi.spyOn(console, 'log').mockImplementation(() => {})
    vi.spyOn(console, 'error').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('renders page title and description', () => {
    render(<ModelEvaluationBenchmarkPage />)

    expect(screen.getByText('模型评估与基准测试')).toBeInTheDocument()
    expect(
      screen.getByText('管理AI模型评估任务，配置基准测试，监控评估进度')
    ).toBeInTheDocument()
  })

  it('displays statistics cards', async () => {
    render(<ModelEvaluationBenchmarkPage />)

    await waitFor(() => {
      expect(screen.getByText('可用基准测试')).toBeInTheDocument()
      expect(screen.getByText('注册模型')).toBeInTheDocument()
      expect(screen.getByText('运行中任务')).toBeInTheDocument()
      expect(screen.getByText('完成任务')).toBeInTheDocument()
    })
  })

  it('renders tabs with correct content', async () => {
    render(<ModelEvaluationBenchmarkPage />)

    // 等待组件加载完成
    await waitFor(() => {
      expect(screen.getByText('评估任务')).toBeInTheDocument()
      expect(screen.getByText('基准测试')).toBeInTheDocument()
      expect(screen.getByText('模型管理')).toBeInTheDocument()
    })

    // 默认显示评估任务tab
    expect(screen.getByText('创建评估任务')).toBeInTheDocument()
  })

  it('opens create evaluation modal when button is clicked', async () => {
    render(<ModelEvaluationBenchmarkPage />)

    await waitFor(() => {
      const createButton = screen.getByText('创建评估任务')
      expect(createButton).toBeInTheDocument()
    })

    // 点击创建评估任务按钮
    await user.click(screen.getByText('创建评估任务'))

    await waitFor(() => {
      expect(screen.getByText('任务名称')).toBeInTheDocument()
      expect(screen.getByText('选择模型')).toBeInTheDocument()
      expect(screen.getByText('选择基准测试')).toBeInTheDocument()
    })
  })

  it('displays benchmarks in benchmark tab', async () => {
    render(<ModelEvaluationBenchmarkPage />)

    // 切换到基准测试tab
    await user.click(screen.getByText('基准测试'))

    await waitFor(() => {
      expect(screen.getByText('CoLA')).toBeInTheDocument()
      expect(screen.getByText('MMLU')).toBeInTheDocument()
      expect(screen.getByText('添加基准测试')).toBeInTheDocument()
    })
  })

  it('displays models in model management tab', async () => {
    render(<ModelEvaluationBenchmarkPage />)

    // 切换到模型管理tab
    await user.click(screen.getByText('模型管理'))

    await waitFor(() => {
      expect(screen.getByText('BERT Base Uncased')).toBeInTheDocument()
      expect(screen.getByText('GPT-3.5 Turbo')).toBeInTheDocument()
      expect(screen.getByText('添加模型')).toBeInTheDocument()
    })
  })

  it('opens add benchmark modal when button is clicked', async () => {
    render(<ModelEvaluationBenchmarkPage />)

    // 切换到基准测试tab
    await user.click(screen.getByText('基准测试'))

    await waitFor(() => {
      expect(screen.getByText('添加基准测试')).toBeInTheDocument()
    })

    // 点击添加基准测试按钮
    await user.click(screen.getByText('添加基准测试'))

    await waitFor(() => {
      expect(screen.getByText('基准测试名称')).toBeInTheDocument()
      expect(screen.getByText('显示名称')).toBeInTheDocument()
      expect(screen.getByText('描述')).toBeInTheDocument()
    })
  })

  it('opens add model modal when button is clicked', async () => {
    render(<ModelEvaluationBenchmarkPage />)

    // 切换到模型管理tab
    await user.click(screen.getByText('模型管理'))

    await waitFor(() => {
      expect(screen.getByText('添加模型')).toBeInTheDocument()
    })

    // 点击添加模型按钮
    await user.click(screen.getByText('添加模型'))

    await waitFor(() => {
      expect(screen.getByText('模型名称')).toBeInTheDocument()
      expect(screen.getByText('模型路径')).toBeInTheDocument()
      expect(screen.getByText('模型类型')).toBeInTheDocument()
    })
  })

  it('shows job details when clicking on job details button', async () => {
    render(<ModelEvaluationBenchmarkPage />)

    await waitFor(() => {
      const detailsButtons = screen.getAllByText('详情')
      expect(detailsButtons.length).toBeGreaterThan(0)
    })

    // 点击第一个详情按钮
    const detailsButtons = screen.getAllByText('详情')
    await user.click(detailsButtons[0])

    await waitFor(() => {
      expect(screen.getByText('评估任务详情')).toBeInTheDocument()
    })
  })

  it('shows evaluation status correctly', async () => {
    render(<ModelEvaluationBenchmarkPage />)

    await waitFor(() => {
      // 检查运行中的任务状态
      expect(screen.getByText('运行中')).toBeInTheDocument()
      // 检查已完成的任务状态
      expect(screen.getByText('已完成')).toBeInTheDocument()
    })
  })

  it('handles refresh button click', async () => {
    const mockConsoleLog = vi.spyOn(console, 'log').mockImplementation(() => {})

    render(<ModelEvaluationBenchmarkPage />)

    await waitFor(() => {
      const refreshButton = screen.getByText('刷新')
      expect(refreshButton).toBeInTheDocument()
    })

    await user.click(screen.getByText('刷新'))

    // 验证数据重新加载
    await waitFor(() => {
      // 数据应该重新显示
      expect(screen.getByText('CoLA')).toBeInTheDocument()
    })

    mockConsoleLog.mockRestore()
  })

  it('displays correct difficulty tags', async () => {
    render(<ModelEvaluationBenchmarkPage />)

    // 切换到基准测试tab
    await user.click(screen.getByText('基准测试'))

    await waitFor(() => {
      // 检查不同难度的标签显示
      const mediumTags = screen.getAllByText('medium')
      const hardTags = screen.getAllByText('hard')

      expect(mediumTags.length).toBeGreaterThan(0)
      expect(hardTags.length).toBeGreaterThan(0)
    })
  })

  it('displays GPU requirement correctly', async () => {
    render(<ModelEvaluationBenchmarkPage />)

    // 切换到基准测试tab
    await user.click(screen.getByText('基准测试'))

    await waitFor(() => {
      // 检查GPU需求标签
      expect(screen.getByText('不需要')).toBeInTheDocument() // CoLA不需要GPU
      expect(screen.getByText('需要')).toBeInTheDocument() // MMLU需要GPU
    })
  })

  it('displays progress bar for running evaluations', async () => {
    render(<ModelEvaluationBenchmarkPage />)

    await waitFor(() => {
      // 检查是否显示进度条
      const progressElements = document.querySelectorAll('.ant-progress')
      expect(progressElements.length).toBeGreaterThan(0)
    })
  })

  it('handles stop evaluation action', async () => {
    render(<ModelEvaluationBenchmarkPage />)

    await waitFor(() => {
      const stopButtons = screen.getAllByText('停止')
      expect(stopButtons.length).toBeGreaterThan(0)
    })

    const stopButton = screen.getAllByText('停止')[0]
    await user.click(stopButton)

    // 验证停止操作被调用
    // 这里可以添加更多的验证逻辑
  })

  it('handles delete evaluation action', async () => {
    render(<ModelEvaluationBenchmarkPage />)

    await waitFor(() => {
      const deleteButtons = screen.getAllByText('删除')
      expect(deleteButtons.length).toBeGreaterThan(0)
    })

    const deleteButton = screen.getAllByText('删除')[0]
    await user.click(deleteButton)

    // 验证删除操作被调用
    // 这里可以添加更多的验证逻辑
  })

  it('displays model parameters count correctly', async () => {
    render(<ModelEvaluationBenchmarkPage />)

    // 切换到模型管理tab
    await user.click(screen.getByText('模型管理'))

    await waitFor(() => {
      // 检查参数数量的显示格式
      expect(screen.getByText('110M')).toBeInTheDocument() // BERT的参数量
      expect(screen.getByText('175000M')).toBeInTheDocument() // GPT的参数量
    })
  })

  it('form validation works correctly', async () => {
    render(<ModelEvaluationBenchmarkPage />)

    // 打开创建评估任务模态框
    await user.click(screen.getByText('创建评估任务'))

    await waitFor(() => {
      expect(screen.getByText('创建并启动')).toBeInTheDocument()
    })

    // 尝试提交空表单
    await user.click(screen.getByText('创建并启动'))

    // 应该显示验证错误
    await waitFor(() => {
      expect(screen.getByText('请输入任务名称')).toBeInTheDocument()
    })
  })

  it('closes modals when cancel button is clicked', async () => {
    render(<ModelEvaluationBenchmarkPage />)

    // 打开创建评估任务模态框
    await user.click(screen.getByText('创建评估任务'))

    await waitFor(() => {
      expect(screen.getByText('取消')).toBeInTheDocument()
    })

    // 点击取消按钮
    await user.click(screen.getByText('取消'))

    await waitFor(() => {
      // 模态框应该关闭，任务名称字段应该不再可见
      expect(screen.queryByText('任务名称')).not.toBeInTheDocument()
    })
  })

  it('displays sample counts in correct format', async () => {
    render(<ModelEvaluationBenchmarkPage />)

    // 切换到基准测试tab
    await user.click(screen.getByText('基准测试'))

    await waitFor(() => {
      // 检查样本数的格式化显示
      expect(screen.getByText('1,043')).toBeInTheDocument() // CoLA的样本数
      expect(screen.getByText('14,042')).toBeInTheDocument() // MMLU的样本数
    })
  })

  it('displays time estimates correctly', async () => {
    render(<ModelEvaluationBenchmarkPage />)

    // 切换到基准测试tab
    await user.click(screen.getByText('基准测试'))

    await waitFor(() => {
      // 检查时间估计的显示
      expect(screen.getByText('5分钟')).toBeInTheDocument() // CoLA的预估时间
      expect(screen.getByText('60分钟')).toBeInTheDocument() // MMLU的预估时间
    })
  })
})

describe('ModelEvaluationBenchmarkPage Form Interactions', () => {
  const user = userEvent.setup()

  beforeEach(() => {
    vi.spyOn(console, 'log').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('allows user to fill evaluation form', async () => {
    render(<ModelEvaluationBenchmarkPage />)

    // 打开创建评估任务模态框
    await user.click(screen.getByText('创建评估任务'))

    await waitFor(() => {
      expect(screen.getByLabelText('任务名称')).toBeInTheDocument()
    })

    // 填写表单
    await user.type(screen.getByLabelText('任务名称'), '测试评估任务')

    // 验证输入
    expect(screen.getByDisplayValue('测试评估任务')).toBeInTheDocument()
  })

  it('allows user to fill benchmark form', async () => {
    render(<ModelEvaluationBenchmarkPage />)

    // 切换到基准测试tab
    await user.click(screen.getByText('基准测试'))

    // 打开添加基准测试模态框
    await user.click(screen.getByText('添加基准测试'))

    await waitFor(() => {
      expect(screen.getByLabelText('基准测试名称')).toBeInTheDocument()
    })

    // 填写表单
    await user.type(screen.getByLabelText('基准测试名称'), 'test-benchmark')
    await user.type(screen.getByLabelText('显示名称'), 'Test Benchmark')

    // 验证输入
    expect(screen.getByDisplayValue('test-benchmark')).toBeInTheDocument()
    expect(screen.getByDisplayValue('Test Benchmark')).toBeInTheDocument()
  })

  it('allows user to fill model form', async () => {
    render(<ModelEvaluationBenchmarkPage />)

    // 切换到模型管理tab
    await user.click(screen.getByText('模型管理'))

    // 打开添加模型模态框
    await user.click(screen.getByText('添加模型'))

    await waitFor(() => {
      expect(screen.getByLabelText('模型名称')).toBeInTheDocument()
    })

    // 填写表单
    await user.type(screen.getByLabelText('模型名称'), '测试模型')
    await user.type(screen.getByLabelText('模型路径'), '/path/to/model')

    // 验证输入
    expect(screen.getByDisplayValue('测试模型')).toBeInTheDocument()
    expect(screen.getByDisplayValue('/path/to/model')).toBeInTheDocument()
  })
})

describe('ModelEvaluationBenchmarkPage Edge Cases', () => {
  beforeEach(() => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('handles loading state correctly', () => {
    render(<ModelEvaluationBenchmarkPage />)

    // 组件应该在初始状态下显示加载中
    expect(document.querySelector('.ant-spin')).toBeInTheDocument()
  })

  it('handles empty data correctly', async () => {
    // 可以通过mock API来测试空数据状态
    render(<ModelEvaluationBenchmarkPage />)

    // 等待加载完成
    await waitFor(() => {
      // 即使没有数据，基本结构也应该存在
      expect(screen.getByText('模型评估与基准测试')).toBeInTheDocument()
    })
  })
})
