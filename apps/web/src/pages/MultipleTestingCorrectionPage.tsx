import React, { useState, useEffect } from 'react';
import { Card, Button, Input, Select, Alert, Tabs, Space, Typography, Row, Col, Form, message, InputNumber, Table, Tag } from 'antd';
import { CalculatorOutlined } from '@ant-design/icons';
import { 
  multipleTestingCorrectionService,
  CorrectionMethod,
  type CorrectionRequest
} from '../services/multipleTestingCorrectionService';

const { Option } = Select;
const { Title, Text } = Typography;

const MultipleTestingCorrectionPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  // P值校正表单
  const [correctionForm, setCorrectionForm] = useState<CorrectionRequest>({
    pvalues: [0.01, 0.02, 0.03, 0.04, 0.05],
    method: CorrectionMethod.BENJAMINI_HOCHBERG,
    alpha: 0.05
  });

  // 方法比较表单
  const [comparisonForm, setComparisonForm] = useState({
    pvalues: [0.01, 0.02, 0.03, 0.04, 0.05],
    alpha: 0.05
  });

  const handleCorrection = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await multipleTestingCorrectionService.correctPValues(correctionForm);
      setResult(response.result);
      setSuccess('P值校正完成');
      message.success('P值校正完成');
    } catch (err) {
      const errorMsg = '校正失败: ' + (err as Error).message;
      setError(errorMsg);
      message.error(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const handleComparison = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await multipleTestingCorrectionService.compareMethods(comparisonForm.pvalues, comparisonForm.alpha);
      setResult(response.result);
      setSuccess('方法比较完成');
      message.success('方法比较完成');
    } catch (err) {
      const errorMsg = '比较失败: ' + (err as Error).message;
      setError(errorMsg);
      message.error(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>多重比较校正</Title>

      {error && (
        <Alert
          message="错误"
          description={error}
          type="error"
          closable
          style={{ marginBottom: 16 }}
          onClose={() => setError(null)}
        />
      )}

      {success && (
        <Alert
          message="成功"
          description={success}
          type="success"
          closable
          style={{ marginBottom: 16 }}
          onClose={() => setSuccess(null)}
        />
      )}

      <Tabs defaultActiveKey="correction" type="card" items={[
        {
          key: 'correction',
          label: 'P值校正',
          children: (
            <Card>
              <Form layout="vertical">
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <Form.Item label="校正方法">
                      <Select 
                        value={correctionForm.method} 
                        onChange={(value) => setCorrectionForm({...correctionForm, method: value as CorrectionMethod})}
                      >
                        <Option value={CorrectionMethod.BONFERRONI}>Bonferroni</Option>
                        <Option value={CorrectionMethod.HOLM}>Holm</Option>
                        <Option value={CorrectionMethod.HOCHBERG}>Hochberg</Option>
                        <Option value={CorrectionMethod.BENJAMINI_HOCHBERG}>Benjamini-Hochberg</Option>
                        <Option value={CorrectionMethod.BENJAMINI_YEKUTIELI}>Benjamini-Yekutieli</Option>
                      </Select>
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item label="显著性水平 (α)">
                      <InputNumber
                        placeholder="显著性水平 (α)"
                        value={correctionForm.alpha}
                        onChange={(value) => setCorrectionForm({...correctionForm, alpha: value || 0.05})}
                        step={0.01}
                        min={0.001}
                        max={0.1}
                        style={{ width: '100%' }}
                      />
                    </Form.Item>
                  </Col>
                  <Col span={24}>
                    <Form.Item label="P值列表 (逗号分隔)">
                      <Input
                        placeholder="例如: 0.01, 0.02, 0.03, 0.04, 0.05"
                        value={correctionForm.pvalues.join(', ')}
                        onChange={(e) => {
                          const values = e.target.value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
                          setCorrectionForm({...correctionForm, pvalues: values});
                        }}
                      />
                    </Form.Item>
                  </Col>
                </Row>

                <Button 
                  type="primary" 
                  icon={<CalculatorOutlined />} 
                  onClick={handleCorrection} 
                  loading={loading}
                >
                  执行P值校正
                </Button>
              </Form>
            </Card>
          )
        },
        {
          key: 'comparison',
          label: '方法比较',
          children: (
            <Card>
              <Form layout="vertical">
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <Form.Item label="显著性水平 (α)">
                      <InputNumber
                        placeholder="显著性水平 (α)"
                        value={comparisonForm.alpha}
                        onChange={(value) => setComparisonForm({...comparisonForm, alpha: value || 0.05})}
                        step={0.01}
                        min={0.001}
                        max={0.1}
                        style={{ width: '100%' }}
                      />
                    </Form.Item>
                  </Col>
                  <Col span={24}>
                    <Form.Item label="P值列表 (逗号分隔)">
                      <Input
                        placeholder="例如: 0.01, 0.02, 0.03, 0.04, 0.05"
                        value={comparisonForm.pvalues.join(', ')}
                        onChange={(e) => {
                          const values = e.target.value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
                          setComparisonForm({...comparisonForm, pvalues: values});
                        }}
                      />
                    </Form.Item>
                  </Col>
                </Row>

                <Button 
                  type="primary" 
                  icon={<CalculatorOutlined />} 
                  onClick={handleComparison} 
                  loading={loading}
                >
                  比较校正方法
                </Button>

                <Card style={{ marginTop: 16, background: '#f0f9ff', border: '1px solid #0ea5e9' }}>
                  <Title level={5}>说明</Title>
                  <ul style={{ marginBottom: 0 }}>
                    <li><strong>Bonferroni</strong>: 最保守的方法，控制家族错误率</li>
                    <li><strong>Holm</strong>: 比Bonferroni稍宽松的逐步方法</li>
                    <li><strong>Hochberg</strong>: 比Holm更宽松的逐步方法</li>
                    <li><strong>Benjamini-Hochberg</strong>: 控制假发现率的经典方法</li>
                    <li><strong>Benjamini-Yekutieli</strong>: 在相关性情况下控制假发现率</li>
                  </ul>
                </Card>
              </Form>
            </Card>
          )
        },
        {
          key: 'results',
          label: '校正结果',
          children: (
            <Card>
              {result ? (
                <div>
                  <Title level={4}>校正结果</Title>
                  <Card style={{ background: '#f6ffed', border: '1px solid #52c41a' }}>
                    <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                      {JSON.stringify(result, null, 2)}
                    </pre>
                  </Card>
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '40px', color: '#999' }}>
                  请先进行校正计算
                </div>
              )}
            </Card>
          )
        }
      ]} />
    </div>
  );
};

export default MultipleTestingCorrectionPage;