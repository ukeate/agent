import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import ConfidenceAnalysis from '../../../src/components/explainer/ConfidenceAnalysis';

describe('ConfidenceAnalysis', () => {
  const mockMetrics = {
    overall_confidence: 0.85,
    prediction_confidence: 0.88,
    evidence_confidence: 0.82,
    model_confidence: 0.86,
    uncertainty_score: 0.15,
    variance: 0.02,
    confidence_interval_lower: 0.78,
    confidence_interval_upper: 0.92,
    confidence_sources: [
      {
        source: '信用评分模型',
        score: 0.90,
        weight: 0.4,
        description: '基于央行征信数据的信用评分',
        reliability: 0.95
      },
      {
        source: '收入验证系统',
        score: 0.85,
        weight: 0.3,
        description: '工资单和银行流水验证',
        reliability: 0.88
      },
      {
        source: '风险评估模型',
        score: 0.80,
        weight: 0.3,
        description: '综合风险因子评估',
        reliability: 0.85
      }
    ],
    calibration_score: 0.87
  };

  const mockUncertaintyFactors = {
    data_quality: 0.12,
    model_complexity: 0.18,
    feature_reliability: 0.10,
    temporal_distance: 0.08,
    context_similarity: 0.15,
    sample_size: 0.14
  };

  it('renders confidence analysis with basic metrics', () => {
    render(<ConfidenceAnalysis metrics={mockMetrics} />);
    
    expect(screen.getByText('置信度分析')).toBeInTheDocument();
    expect(screen.getByText('85%')).toBeInTheDocument(); // overall confidence
    expect(screen.getByText('15%')).toBeInTheDocument(); // uncertainty score
    expect(screen.getByText('[78%, 92%]')).toBeInTheDocument(); // confidence interval
  });

  it('displays confidence level badges correctly', () => {
    render(<ConfidenceAnalysis metrics={mockMetrics} />);
    
    expect(screen.getByText('高')).toBeInTheDocument(); // 85% should be "高"
  });

  it('shows all tab options', () => {
    render(<ConfidenceAnalysis metrics={mockMetrics} />);
    
    expect(screen.getByText('概览')).toBeInTheDocument();
    expect(screen.getByText('置信度分解')).toBeInTheDocument();
    expect(screen.getByText('不确定性分析')).toBeInTheDocument();
    expect(screen.getByText('校准分析')).toBeInTheDocument();
  });

  it('switches between tabs correctly', async () => {
    render(<ConfidenceAnalysis metrics={mockMetrics} />);
    
    // 切换到置信度分解
    fireEvent.click(screen.getByText('置信度分解'));
    
    await waitFor(() => {
      expect(screen.getByText('置信度来源分解')).toBeInTheDocument();
      expect(screen.getByText('信用评分模型')).toBeInTheDocument();
      expect(screen.getByText('收入验证系统')).toBeInTheDocument();
      expect(screen.getByText('风险评估模型')).toBeInTheDocument();
    });
  });

  it('displays confidence sources with correct data', async () => {
    render(<ConfidenceAnalysis metrics={mockMetrics} />);
    
    fireEvent.click(screen.getByText('置信度分解'));
    
    await waitFor(() => {
      expect(screen.getByText('90.0%')).toBeInTheDocument(); // 信用评分模型分数
      expect(screen.getByText('85.0%')).toBeInTheDocument(); // 收入验证系统分数
      expect(screen.getByText('80.0%')).toBeInTheDocument(); // 风险评估模型分数
      
      expect(screen.getByText('权重: 0.4')).toBeInTheDocument();
      expect(screen.getByText('权重: 0.3')).toBeInTheDocument();
      
      expect(screen.getByText('可靠性: 95%')).toBeInTheDocument();
      expect(screen.getByText('可靠性: 88%')).toBeInTheDocument();
    });
  });

  it('shows uncertainty factors analysis', async () => {
    render(<ConfidenceAnalysis metrics={mockMetrics} uncertaintyFactors={mockUncertaintyFactors} />);
    
    fireEvent.click(screen.getByText('不确定性分析'));
    
    await waitFor(() => {
      expect(screen.getByText('不确定性因子分析')).toBeInTheDocument();
      expect(screen.getByText('数据质量')).toBeInTheDocument();
      expect(screen.getByText('模型复杂度')).toBeInTheDocument();
      expect(screen.getByText('特征可靠性')).toBeInTheDocument();
      expect(screen.getByText('时间距离')).toBeInTheDocument();
      expect(screen.getByText('上下文相似性')).toBeInTheDocument();
      expect(screen.getByText('样本大小')).toBeInTheDocument();
    });
  });

  it('displays calibration analysis when available', async () => {
    render(<ConfidenceAnalysis metrics={mockMetrics} />);
    
    fireEvent.click(screen.getByText('校准分析'));
    
    await waitFor(() => {
      expect(screen.getByText('置信度校准分析')).toBeInTheDocument();
      expect(screen.getByText('校准分数')).toBeInTheDocument();
      expect(screen.getByText('87.0%')).toBeInTheDocument(); // calibration score
      expect(screen.getByText('校准质量良好')).toBeInTheDocument();
    });
  });

  it('shows confidence interpretation in overview', () => {
    render(<ConfidenceAnalysis metrics={mockMetrics} />);
    
    expect(screen.getByText('置信度解读')).toBeInTheDocument();
    expect(screen.getByText('高置信度 (85.0%)')).toBeInTheDocument();
    expect(screen.getByText('决策具有高可信度，建议采用并保持监控。')).toBeInTheDocument();
  });

  it('displays key metrics in overview', () => {
    render(<ConfidenceAnalysis metrics={mockMetrics} />);
    
    expect(screen.getByText('关键指标')).toBeInTheDocument();
    expect(screen.getByText('预测置信度:')).toBeInTheDocument();
    expect(screen.getByText('88.0%')).toBeInTheDocument();
    expect(screen.getByText('证据置信度:')).toBeInTheDocument();
    expect(screen.getByText('82.0%')).toBeInTheDocument();
    expect(screen.getByText('模型置信度:')).toBeInTheDocument();
    expect(screen.getByText('86.0%')).toBeInTheDocument();
  });

  it('shows risk assessment based on uncertainty', () => {
    render(<ConfidenceAnalysis metrics={mockMetrics} />);
    
    expect(screen.getByText('风险评估')).toBeInTheDocument();
    expect(screen.getByText('低风险决策')).toBeInTheDocument(); // uncertainty 0.15 <= 0.2
  });

  it('handles high uncertainty correctly', () => {
    const highUncertaintyMetrics = {
      ...mockMetrics,
      uncertainty_score: 0.5,
      overall_confidence: 0.5
    };
    
    render(<ConfidenceAnalysis metrics={highUncertaintyMetrics} />);
    
    expect(screen.getByText('高风险，需要谨慎处理')).toBeInTheDocument();
    expect(screen.getByText('决策可信度较低，建议重新评估或收集更多高质量数据。')).toBeInTheDocument();
  });

  it('calculates weighted contributions correctly', async () => {
    render(<ConfidenceAnalysis metrics={mockMetrics} />);
    
    fireEvent.click(screen.getByText('置信度分解'));
    
    await waitFor(() => {
      // 信用评分模型: 0.90 * 0.4 = 0.36 = 36.0%
      expect(screen.getByText('加权贡献: 36.0%')).toBeInTheDocument();
      // 收入验证系统: 0.85 * 0.3 = 0.255 = 25.5%
      expect(screen.getByText('加权贡献: 25.5%')).toBeInTheDocument();
      // 风险评估模型: 0.80 * 0.3 = 0.24 = 24.0%
      expect(screen.getByText('加权贡献: 24.0%')).toBeInTheDocument();
    });
  });

  it('shows uncertainty factor descriptions', async () => {
    render(<ConfidenceAnalysis metrics={mockMetrics} uncertaintyFactors={mockUncertaintyFactors} />);
    
    fireEvent.click(screen.getByText('不确定性分析'));
    
    await waitFor(() => {
      expect(screen.getByText('输入数据的质量和完整性影响')).toBeInTheDocument();
      expect(screen.getByText('模型复杂度带来的不确定性')).toBeInTheDocument();
      expect(screen.getByText('特征变量的可靠性评估')).toBeInTheDocument();
      expect(screen.getByText('时间距离对预测的影响')).toBeInTheDocument();
      expect(screen.getByText('上下文匹配度的影响')).toBeInTheDocument();
      expect(screen.getByText('样本数量对结果的影响')).toBeInTheDocument();
    });
  });

  it('provides calibration recommendations', async () => {
    render(<ConfidenceAnalysis metrics={mockMetrics} />);
    
    fireEvent.click(screen.getByText('校准分析'));
    
    await waitFor(() => {
      expect(screen.getByText('校准建议')).toBeInTheDocument();
      expect(screen.getByText('收集更多历史决策数据以改进校准')).toBeInTheDocument();
      expect(screen.getByText('定期评估预测准确性与置信度的匹配程度')).toBeInTheDocument();
      expect(screen.getByText('考虑不同场景下的校准表现差异')).toBeInTheDocument();
      expect(screen.getByText('建立置信度校准的反馈机制')).toBeInTheDocument();
    });
  });

  it('handles missing optional fields gracefully', () => {
    const minimalMetrics = {
      overall_confidence: 0.75,
      uncertainty_score: 0.25,
      confidence_sources: []
    };
    
    render(<ConfidenceAnalysis metrics={minimalMetrics} />);
    
    expect(screen.getByText('75%')).toBeInTheDocument();
    expect(screen.getByText('25%')).toBeInTheDocument();
    // 不应该崩溃
  });

  it('applies correct confidence level styling', () => {
    // 测试极高置信度
    const veryHighMetrics = { ...mockMetrics, overall_confidence: 0.95 };
    const { rerender } = render(<ConfidenceAnalysis metrics={veryHighMetrics} />);
    expect(screen.getByText('极高')).toBeInTheDocument();
    
    // 测试低置信度
    const lowMetrics = { ...mockMetrics, overall_confidence: 0.45 };
    rerender(<ConfidenceAnalysis metrics={lowMetrics} />);
    expect(screen.getByText('中等偏低')).toBeInTheDocument();
  });

  it('uses default uncertainty factors when not provided', async () => {
    render(<ConfidenceAnalysis metrics={mockMetrics} />);
    
    fireEvent.click(screen.getByText('不确定性分析'));
    
    await waitFor(() => {
      // 应该显示默认的不确定性因子
      expect(screen.getByText('数据质量')).toBeInTheDocument();
      expect(screen.getByText('模型复杂度')).toBeInTheDocument();
    });
  });
});