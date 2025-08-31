/**
 * 参数配置组件单元测试
 */
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import ParameterConfig from '../ParameterConfig';

const mockInitialConfig = {
  learning_rate: {
    type: 'float',
    low: 0.001,
    high: 0.1,
    log: true
  },
  batch_size: {
    type: 'int',
    low: 16,
    high: 256,
    step: 16
  },
  optimizer: {
    type: 'categorical',
    choices: ['adam', 'sgd', 'rmsprop']
  }
};

describe('ParameterConfig', () => {
  const mockOnChange = vi.fn();
  
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('渲染参数配置表单', () => {
    render(
      <ParameterConfig 
        config={mockInitialConfig}
        onChange={mockOnChange}
      />
    );

    expect(screen.getByText('learning_rate')).toBeInTheDocument();
    expect(screen.getByText('batch_size')).toBeInTheDocument();
    expect(screen.getByText('optimizer')).toBeInTheDocument();
  });

  it('显示正确的参数类型', () => {
    render(
      <ParameterConfig 
        config={mockInitialConfig}
        onChange={mockOnChange}
      />
    );

    expect(screen.getByDisplayValue('float')).toBeInTheDocument();
    expect(screen.getByDisplayValue('int')).toBeInTheDocument();
    expect(screen.getByDisplayValue('categorical')).toBeInTheDocument();
  });

  it('显示数值参数的范围输入', () => {
    render(
      <ParameterConfig 
        config={mockInitialConfig}
        onChange={mockOnChange}
      />
    );

    // learning_rate的范围
    expect(screen.getByDisplayValue('0.001')).toBeInTheDocument();
    expect(screen.getByDisplayValue('0.1')).toBeInTheDocument();
    
    // batch_size的范围
    expect(screen.getByDisplayValue('16')).toBeInTheDocument();
    expect(screen.getByDisplayValue('256')).toBeInTheDocument();
  });

  it('显示分类参数的选项', () => {
    render(
      <ParameterConfig 
        config={mockInitialConfig}
        onChange={mockOnChange}
      />
    );

    expect(screen.getByDisplayValue('adam,sgd,rmsprop')).toBeInTheDocument();
  });

  it('支持添加新参数', async () => {
    render(
      <ParameterConfig 
        config={mockInitialConfig}
        onChange={mockOnChange}
      />
    );

    const addButton = screen.getByRole('button', { name: /添加参数/ });
    fireEvent.click(addButton);

    // 应该出现新参数的表单
    const nameInputs = screen.getAllByPlaceholderText(/参数名/);
    expect(nameInputs.length).toBeGreaterThan(0);

    // 输入新参数信息
    const lastNameInput = nameInputs[nameInputs.length - 1];
    fireEvent.change(lastNameInput, { target: { value: 'dropout_rate' } });

    // 选择参数类型
    const typeSelects = screen.getAllByDisplayValue('float');
    const lastTypeSelect = typeSelects[typeSelects.length - 1];
    fireEvent.change(lastTypeSelect, { target: { value: 'float' } });

    // 触发onChange
    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalled();
    });
  });

  it('支持删除参数', async () => {
    render(
      <ParameterConfig 
        config={mockInitialConfig}
        onChange={mockOnChange}
      />
    );

    // 找到删除按钮
    const deleteButtons = screen.getAllByRole('button', { name: /删除/ });
    expect(deleteButtons.length).toBeGreaterThan(0);

    fireEvent.click(deleteButtons[0]);

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalled();
    });
  });

  it('支持修改参数类型', async () => {
    render(
      <ParameterConfig 
        config={mockInitialConfig}
        onChange={mockOnChange}
      />
    );

    // 修改第一个参数的类型
    const typeSelect = screen.getByDisplayValue('float');
    fireEvent.change(typeSelect, { target: { value: 'int' } });

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalled();
    });
  });

  it('支持修改数值范围', async () => {
    render(
      <ParameterConfig 
        config={mockInitialConfig}
        onChange={mockOnChange}
      />
    );

    // 修改learning_rate的最小值
    const lowInput = screen.getByDisplayValue('0.001');
    fireEvent.change(lowInput, { target: { value: '0.0001' } });

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalled();
    });
  });

  it('支持修改分类选项', async () => {
    render(
      <ParameterConfig 
        config={mockInitialConfig}
        onChange={mockOnChange}
      />
    );

    // 修改optimizer的选项
    const choicesInput = screen.getByDisplayValue('adam,sgd,rmsprop');
    fireEvent.change(choicesInput, { target: { value: 'adam,sgd,adamw' } });

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalled();
    });
  });

  it('验证参数名唯一性', async () => {
    render(
      <ParameterConfig 
        config={mockInitialConfig}
        onChange={mockOnChange}
      />
    );

    // 添加新参数
    const addButton = screen.getByRole('button', { name: /添加参数/ });
    fireEvent.click(addButton);

    // 输入已存在的参数名
    const nameInputs = screen.getAllByPlaceholderText(/参数名/);
    const newNameInput = nameInputs[nameInputs.length - 1];
    fireEvent.change(newNameInput, { target: { value: 'learning_rate' } });

    // 应该显示错误信息
    await waitFor(() => {
      expect(screen.getByText(/参数名已存在/)).toBeInTheDocument();
    });
  });

  it('验证数值范围有效性', async () => {
    render(
      <ParameterConfig 
        config={mockInitialConfig}
        onChange={mockOnChange}
      />
    );

    // 设置无效范围（最小值大于最大值）
    const lowInput = screen.getByDisplayValue('0.001');
    fireEvent.change(lowInput, { target: { value: '0.2' } });

    // 应该显示错误信息
    await waitFor(() => {
      expect(screen.getByText(/最小值必须小于最大值/)).toBeInTheDocument();
    });
  });

  it('支持对数刻度设置', () => {
    render(
      <ParameterConfig 
        config={mockInitialConfig}
        onChange={mockOnChange}
      />
    );

    // learning_rate应该显示对数刻度选项
    const logCheckbox = screen.getByRole('checkbox', { name: /对数刻度/ });
    expect(logCheckbox).toBeChecked();
  });

  it('支持步长设置', () => {
    render(
      <ParameterConfig 
        config={mockInitialConfig}
        onChange={mockOnChange}
      />
    );

    // batch_size应该显示步长输入
    const stepInput = screen.getByDisplayValue('16');
    expect(stepInput).toBeInTheDocument();
  });

  it('处理空配置', () => {
    render(
      <ParameterConfig 
        config={{}}
        onChange={mockOnChange}
      />
    );

    expect(screen.getByText(/暂无参数配置/)).toBeInTheDocument();
  });

  it('支持从JSON导入', async () => {
    render(
      <ParameterConfig 
        config={{}}
        onChange={mockOnChange}
      />
    );

    const importButton = screen.getByRole('button', { name: /从JSON导入/ });
    fireEvent.click(importButton);

    // 应该出现JSON输入框
    const jsonTextarea = screen.getByPlaceholderText(/输入JSON配置/);
    expect(jsonTextarea).toBeInTheDocument();

    // 输入有效JSON
    const validJson = JSON.stringify(mockInitialConfig);
    fireEvent.change(jsonTextarea, { target: { value: validJson } });

    // 确认导入
    const confirmButton = screen.getByRole('button', { name: /确认导入/ });
    fireEvent.click(confirmButton);

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalledWith(mockInitialConfig);
    });
  });

  it('处理无效JSON导入', async () => {
    render(
      <ParameterConfig 
        config={{}}
        onChange={mockOnChange}
      />
    );

    const importButton = screen.getByRole('button', { name: /从JSON导入/ });
    fireEvent.click(importButton);

    const jsonTextarea = screen.getByPlaceholderText(/输入JSON配置/);
    fireEvent.change(jsonTextarea, { target: { value: 'invalid json' } });

    const confirmButton = screen.getByRole('button', { name: /确认导入/ });
    fireEvent.click(confirmButton);

    await waitFor(() => {
      expect(screen.getByText(/JSON格式无效/)).toBeInTheDocument();
    });
  });

  it('支持导出为JSON', () => {
    render(
      <ParameterConfig 
        config={mockInitialConfig}
        onChange={mockOnChange}
      />
    );

    const exportButton = screen.getByRole('button', { name: /导出JSON/ });
    fireEvent.click(exportButton);

    // 应该显示JSON代码
    const jsonCode = screen.getByText(/"learning_rate"/);
    expect(jsonCode).toBeInTheDocument();
  });

  it('支持参数模板', async () => {
    render(
      <ParameterConfig 
        config={{}}
        onChange={mockOnChange}
      />
    );

    const templateButton = screen.getByRole('button', { name: /使用模板/ });
    fireEvent.click(templateButton);

    // 选择深度学习模板
    const deepLearningTemplate = screen.getByText(/深度学习/);
    fireEvent.click(deepLearningTemplate);

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalled();
    });
  });

  it('显示参数统计信息', () => {
    render(
      <ParameterConfig 
        config={mockInitialConfig}
        onChange={mockOnChange}
      />
    );

    // 应该显示参数数量
    expect(screen.getByText(/参数总数: 3/)).toBeInTheDocument();
    expect(screen.getByText(/数值参数: 2/)).toBeInTheDocument();
    expect(screen.getByText(/分类参数: 1/)).toBeInTheDocument();
  });

  it('支持参数分组', () => {
    const groupedConfig = {
      model: {
        learning_rate: mockInitialConfig.learning_rate,
        batch_size: mockInitialConfig.batch_size
      },
      optimizer: {
        optimizer: mockInitialConfig.optimizer
      }
    };

    render(
      <ParameterConfig 
        config={groupedConfig}
        onChange={mockOnChange}
        enableGrouping
      />
    );

    expect(screen.getByText('model')).toBeInTheDocument();
    expect(screen.getByText('optimizer')).toBeInTheDocument();
  });

  it('支持条件参数', () => {
    const conditionalConfig = {
      optimizer: mockInitialConfig.optimizer,
      learning_rate: {
        ...mockInitialConfig.learning_rate,
        condition: { optimizer: 'adam' }
      }
    };

    render(
      <ParameterConfig 
        config={conditionalConfig}
        onChange={mockOnChange}
        enableConditionalParams
      />
    );

    expect(screen.getByText(/条件:/)).toBeInTheDocument();
  });
});