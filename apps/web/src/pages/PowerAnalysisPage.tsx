import React, { useState, useEffect } from 'react'
import {
  Card,
  Button,
  Input,
  Select,
  Alert,
  Tabs,
  Space,
  Typography,
  Row,
  Col,
  Form,
  message,
  InputNumber,
} from 'antd'
import { CalculatorOutlined } from '@ant-design/icons'
import {
  powerAnalysisService,
  TestType,
  AlternativeHypothesis,
  type PowerCalculationRequest,
} from '../services/powerAnalysisService'

const { Option } = Select
const { Title, Text } = Typography

const PowerAnalysisPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [result, setResult] = useState<any>(null)

  // 功效计算表单
  const [powerForm, setPowerForm] = useState<PowerCalculationRequest>({
    test_type: TestType.TWO_SAMPLE_T,
    effect_size: 0.5,
    sample_size: 30,
    alpha: 0.05,
    alternative: AlternativeHypothesis.TWO_SIDED,
  })

  // 样本量计算表单
  const [sampleSizeForm, setSampleSizeForm] = useState({
    test_type: TestType.TWO_SAMPLE_T,
    effect_size: 0.5,
    power: 0.8,
    alpha: 0.05,
    alternative: AlternativeHypothesis.TWO_SIDED,
    ratio: 1.0,
  })

  // A/B测试样本量计算表单
  const [abTestForm, setAbTestForm] = useState({
    baseline_conversion_rate: 0.1,
    minimum_detectable_effect: 0.1,
    power: 0.8,
    alpha: 0.05,
    alternative: AlternativeHypothesis.TWO_SIDED,
  })

  const handlePowerCalculation = async () => {
    try {
      setLoading(true)
      setError(null)

      const response = await powerAnalysisService.calculatePower(powerForm)
      setResult(response.result)
      setSuccess('功效计算完成')
      message.success('功效计算完成')
    } catch (err) {
      const errorMsg = '计算失败: ' + (err as Error).message
      setError(errorMsg)
      message.error(errorMsg)
    } finally {
      setLoading(false)
    }
  }

  const handleSampleSizeCalculation = async () => {
    try {
      setLoading(true)
      setError(null)

      const response =
        await powerAnalysisService.calculateSampleSize(sampleSizeForm)
      setResult(response.result)
      setSuccess('样本量计算完成')
      message.success('样本量计算完成')
    } catch (err) {
      const errorMsg = '计算失败: ' + (err as Error).message
      setError(errorMsg)
      message.error(errorMsg)
    } finally {
      setLoading(false)
    }
  }

  const handleABTestCalculation = async () => {
    try {
      setLoading(true)
      setError(null)

      const response =
        await powerAnalysisService.calculateABTestSampleSize(abTestForm)
      setResult(response.result)
      setSuccess('A/B测试样本量计算完成')
      message.success('A/B测试样本量计算完成')
    } catch (err) {
      const errorMsg = '计算失败: ' + (err as Error).message
      setError(errorMsg)
      message.error(errorMsg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>统计功效分析</Title>

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

      <Tabs
        defaultActiveKey="power"
        type="card"
        items={[
          {
            key: 'power',
            label: '功效计算',
            children: (
              <Card>
                <Form layout="vertical">
                  <Row gutter={[16, 16]}>
                    <Col span={12}>
                      <Form.Item label="检验类型">
                        <Select
                          value={powerForm.test_type}
                          onChange={value =>
                            setPowerForm({
                              ...powerForm,
                              test_type: value as TestType,
                            })
                          }
                        >
                          <Option value={TestType.ONE_SAMPLE_T}>
                            单样本t检验
                          </Option>
                          <Option value={TestType.TWO_SAMPLE_T}>
                            双样本t检验
                          </Option>
                          <Option value={TestType.PAIRED_T}>配对t检验</Option>
                          <Option value={TestType.ONE_PROPORTION}>
                            单比例检验
                          </Option>
                          <Option value={TestType.TWO_PROPORTIONS}>
                            双比例检验
                          </Option>
                        </Select>
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="效应量">
                        <InputNumber
                          placeholder="效应量"
                          value={powerForm.effect_size}
                          onChange={value =>
                            setPowerForm({
                              ...powerForm,
                              effect_size: value || 0.5,
                            })
                          }
                          step={0.1}
                          style={{ width: '100%' }}
                        />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="样本量">
                        <InputNumber
                          placeholder="样本量"
                          value={powerForm.sample_size}
                          onChange={value =>
                            setPowerForm({
                              ...powerForm,
                              sample_size: value || 30,
                            })
                          }
                          style={{ width: '100%' }}
                        />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="显著性水平 (α)">
                        <InputNumber
                          placeholder="显著性水平 (α)"
                          value={powerForm.alpha}
                          onChange={value =>
                            setPowerForm({ ...powerForm, alpha: value || 0.05 })
                          }
                          step={0.01}
                          min={0.001}
                          max={0.1}
                          style={{ width: '100%' }}
                        />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="假设检验类型">
                        <Select
                          value={powerForm.alternative}
                          onChange={value =>
                            setPowerForm({
                              ...powerForm,
                              alternative: value as AlternativeHypothesis,
                            })
                          }
                        >
                          <Option value={AlternativeHypothesis.TWO_SIDED}>
                            双侧检验
                          </Option>
                          <Option value={AlternativeHypothesis.GREATER}>
                            右侧检验
                          </Option>
                          <Option value={AlternativeHypothesis.LESS}>
                            左侧检验
                          </Option>
                        </Select>
                      </Form.Item>
                    </Col>
                  </Row>

                  <Button
                    type="primary"
                    icon={<CalculatorOutlined />}
                    onClick={handlePowerCalculation}
                    loading={loading}
                  >
                    计算统计功效
                  </Button>
                </Form>
              </Card>
            ),
          },
          {
            key: 'sample-size',
            label: '样本量计算',
            children: (
              <Card>
                <Form layout="vertical">
                  <Row gutter={[16, 16]}>
                    <Col span={12}>
                      <Form.Item label="检验类型">
                        <Select
                          value={sampleSizeForm.test_type}
                          onChange={value =>
                            setSampleSizeForm({
                              ...sampleSizeForm,
                              test_type: value as TestType,
                            })
                          }
                        >
                          <Option value={TestType.ONE_SAMPLE_T}>
                            单样本t检验
                          </Option>
                          <Option value={TestType.TWO_SAMPLE_T}>
                            双样本t检验
                          </Option>
                          <Option value={TestType.PAIRED_T}>配对t检验</Option>
                          <Option value={TestType.ONE_PROPORTION}>
                            单比例检验
                          </Option>
                          <Option value={TestType.TWO_PROPORTIONS}>
                            双比例检验
                          </Option>
                        </Select>
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="效应量">
                        <InputNumber
                          placeholder="效应量"
                          value={sampleSizeForm.effect_size}
                          onChange={value =>
                            setSampleSizeForm({
                              ...sampleSizeForm,
                              effect_size: value || 0.5,
                            })
                          }
                          step={0.1}
                          style={{ width: '100%' }}
                        />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="期望统计功效">
                        <InputNumber
                          placeholder="期望统计功效"
                          value={sampleSizeForm.power}
                          onChange={value =>
                            setSampleSizeForm({
                              ...sampleSizeForm,
                              power: value || 0.8,
                            })
                          }
                          step={0.01}
                          min={0.5}
                          max={0.99}
                          style={{ width: '100%' }}
                        />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="显著性水平 (α)">
                        <InputNumber
                          placeholder="显著性水平 (α)"
                          value={sampleSizeForm.alpha}
                          onChange={value =>
                            setSampleSizeForm({
                              ...sampleSizeForm,
                              alpha: value || 0.05,
                            })
                          }
                          step={0.01}
                          min={0.001}
                          max={0.1}
                          style={{ width: '100%' }}
                        />
                      </Form.Item>
                    </Col>
                  </Row>

                  <Button
                    type="primary"
                    icon={<CalculatorOutlined />}
                    onClick={handleSampleSizeCalculation}
                    loading={loading}
                  >
                    计算所需样本量
                  </Button>
                </Form>
              </Card>
            ),
          },
          {
            key: 'ab-test',
            label: 'A/B测试样本量',
            children: (
              <Card>
                <Form layout="vertical">
                  <Row gutter={[16, 16]}>
                    <Col span={12}>
                      <Form.Item label="基准转化率">
                        <InputNumber
                          placeholder="基准转化率"
                          value={abTestForm.baseline_conversion_rate}
                          onChange={value =>
                            setAbTestForm({
                              ...abTestForm,
                              baseline_conversion_rate: value || 0.1,
                            })
                          }
                          step={0.001}
                          min={0}
                          max={1}
                          style={{ width: '100%' }}
                        />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="最小可检测效应">
                        <InputNumber
                          placeholder="最小可检测效应"
                          value={abTestForm.minimum_detectable_effect}
                          onChange={value =>
                            setAbTestForm({
                              ...abTestForm,
                              minimum_detectable_effect: value || 0.1,
                            })
                          }
                          step={0.001}
                          min={0}
                          max={1}
                          style={{ width: '100%' }}
                        />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="期望统计功效">
                        <InputNumber
                          placeholder="期望统计功效"
                          value={abTestForm.power}
                          onChange={value =>
                            setAbTestForm({
                              ...abTestForm,
                              power: value || 0.8,
                            })
                          }
                          step={0.01}
                          min={0.5}
                          max={0.99}
                          style={{ width: '100%' }}
                        />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="显著性水平 (α)">
                        <InputNumber
                          placeholder="显著性水平 (α)"
                          value={abTestForm.alpha}
                          onChange={value =>
                            setAbTestForm({
                              ...abTestForm,
                              alpha: value || 0.05,
                            })
                          }
                          step={0.01}
                          min={0.001}
                          max={0.1}
                          style={{ width: '100%' }}
                        />
                      </Form.Item>
                    </Col>
                  </Row>

                  <Button
                    type="primary"
                    icon={<CalculatorOutlined />}
                    onClick={handleABTestCalculation}
                    loading={loading}
                  >
                    计算A/B测试样本量
                  </Button>

                  <Card
                    style={{
                      marginTop: 16,
                      background: '#f0f9ff',
                      border: '1px solid #0ea5e9',
                    }}
                  >
                    <Title level={5}>说明</Title>
                    <ul style={{ marginBottom: 0 }}>
                      <li>基准转化率: 当前版本的转化率</li>
                      <li>最小可检测效应: 希望检测到的最小相对提升</li>
                      <li>统计功效: 检测到真实效应的概率，通常设为0.8</li>
                      <li>显著性水平: 第一类错误概率，通常设为0.05</li>
                    </ul>
                  </Card>
                </Form>
              </Card>
            ),
          },
          {
            key: 'results',
            label: '计算结果',
            children: (
              <Card>
                {result ? (
                  <div>
                    <Title level={4}>计算结果</Title>
                    <Card
                      style={{
                        background: '#f6ffed',
                        border: '1px solid #52c41a',
                      }}
                    >
                      <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                        {JSON.stringify(result, null, 2)}
                      </pre>
                    </Card>
                  </div>
                ) : (
                  <div
                    style={{
                      textAlign: 'center',
                      padding: '40px',
                      color: '#999',
                    }}
                  >
                    请先进行计算
                  </div>
                )}
              </Card>
            ),
          },
        ]}
      />
    </div>
  )
}

export default PowerAnalysisPage
