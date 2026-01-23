/**
 * 参数配置组件
 */
import React, { useState, useEffect } from 'react'
import {
  Form,
  Input,
  Select,
  InputNumber,
  Button,
  Space,
  Card,
  Row,
  Col,
  Checkbox,
  Modal,
  message,
  Empty,
  Statistic,
} from 'antd'
import {
  PlusOutlined,
  DeleteOutlined,
  UploadOutlined,
  DownloadOutlined,
  CopyOutlined,
} from '@ant-design/icons'

const { Option } = Select
const { TextArea } = Input

interface ParameterRange {
  type: 'float' | 'int' | 'categorical'
  low?: number
  high?: number
  log?: boolean
  step?: number
  choices?: string[]
  condition?: any
}

interface ParameterConfigProps {
  config: { [key: string]: ParameterRange | any }
  onChange: (config: any) => void
  enableGrouping?: boolean
  enableConditionalParams?: boolean
}

const ParameterConfig: React.FC<ParameterConfigProps> = ({
  config,
  onChange,
  enableGrouping = false,
  enableConditionalParams = false,
}) => {
  const [parameters, setParameters] = useState<{
    [key: string]: ParameterRange
  }>({})
  const [newParamName, setNewParamName] = useState('')
  const [newParamType, setNewParamType] = useState<
    'float' | 'int' | 'categorical'
  >('float')
  const [importModalVisible, setImportModalVisible] = useState(false)
  const [exportModalVisible, setExportModalVisible] = useState(false)
  const [jsonInput, setJsonInput] = useState('')
  const [errors, setErrors] = useState<{ [key: string]: string }>({})

  useEffect(() => {
    // 处理分组配置
    if (enableGrouping && typeof config === 'object') {
      const flatParams: { [key: string]: ParameterRange } = {}
      Object.entries(config).forEach(([group, params]) => {
        if (typeof params === 'object') {
          Object.entries(params as any).forEach(([key, value]) => {
            flatParams[key] = value as ParameterRange
          })
        }
      })
      setParameters(flatParams)
    } else {
      setParameters(config as { [key: string]: ParameterRange })
    }
  }, [config, enableGrouping])

  const handleAddParameter = () => {
    if (!newParamName) {
      message.error('请输入参数名')
      return
    }

    if (parameters[newParamName]) {
      setErrors({ ...errors, [newParamName]: '参数名已存在' })
      return
    }

    const newParam: ParameterRange = {
      type: newParamType,
      ...(newParamType === 'float' ? { low: 0, high: 1, log: false } : {}),
      ...(newParamType === 'int' ? { low: 0, high: 100, step: 1 } : {}),
      ...(newParamType === 'categorical'
        ? { choices: ['option1', 'option2'] }
        : {}),
    }

    const updatedParams = { ...parameters, [newParamName]: newParam }
    setParameters(updatedParams)
    onChange(updatedParams)
    setNewParamName('')
    setErrors({})
  }

  const handleDeleteParameter = (key: string) => {
    const updatedParams = { ...parameters }
    delete updatedParams[key]
    setParameters(updatedParams)
    onChange(updatedParams)
  }

  const handleParameterChange = (key: string, field: string, value: any) => {
    const updatedParams = {
      ...parameters,
      [key]: {
        ...parameters[key],
        [field]: value,
      },
    }

    // 验证范围
    if (field === 'low' || field === 'high') {
      const param = updatedParams[key]
      if (
        param.low !== undefined &&
        param.high !== undefined &&
        param.low > param.high
      ) {
        setErrors({ ...errors, [key]: '最小值必须小于最大值' })
        return
      } else {
        const newErrors = { ...errors }
        delete newErrors[key]
        setErrors(newErrors)
      }
    }

    setParameters(updatedParams)
    onChange(updatedParams)
  }

  const handleImportJSON = () => {
    try {
      const imported = JSON.parse(jsonInput)
      setParameters(imported)
      onChange(imported)
      setImportModalVisible(false)
      setJsonInput('')
      message.success('导入成功')
    } catch (error) {
      message.error('JSON格式无效')
    }
  }

  const handleExportJSON = () => {
    setExportModalVisible(true)
  }

  const handleUseTemplate = (templateName: string) => {
    const templates: { [key: string]: any } = {
      深度学习: {
        learning_rate: { type: 'float', low: 0.0001, high: 0.1, log: true },
        batch_size: { type: 'int', low: 16, high: 256, step: 16 },
        dropout_rate: { type: 'float', low: 0.0, high: 0.5 },
        optimizer: { type: 'categorical', choices: ['adam', 'sgd', 'rmsprop'] },
      },
      机器学习: {
        n_estimators: { type: 'int', low: 10, high: 500, step: 10 },
        max_depth: { type: 'int', low: 3, high: 20 },
        min_samples_split: { type: 'int', low: 2, high: 20 },
        criterion: { type: 'categorical', choices: ['gini', 'entropy'] },
      },
    }

    const template = templates[templateName]
    if (template) {
      setParameters(template)
      onChange(template)
      message.success(`已应用${templateName}模板`)
    }
  }

  const parameterKeys = Object.keys(parameters)

  if (parameterKeys.length === 0) {
    return (
      <div>
        <Empty description="暂无参数配置">
          <Space>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => setNewParamName('new_param')}
            >
              添加参数
            </Button>
            <Button
              icon={<UploadOutlined />}
              onClick={() => setImportModalVisible(true)}
            >
              从JSON导入
            </Button>
            <Button onClick={() => handleUseTemplate('深度学习')}>
              使用模板
            </Button>
          </Space>
        </Empty>
      </div>
    )
  }

  // 计算统计信息
  const stats = {
    total: parameterKeys.length,
    numeric: parameterKeys.filter(k =>
      ['float', 'int'].includes(parameters[k].type)
    ).length,
    categorical: parameterKeys.filter(k => parameters[k].type === 'categorical')
      .length,
  }

  return (
    <div>
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={8}>
          <Statistic title="参数总数" value={`参数总数: ${stats.total}`} />
        </Col>
        <Col span={8}>
          <Statistic title="数值参数" value={`数值参数: ${stats.numeric}`} />
        </Col>
        <Col span={8}>
          <Statistic
            title="分类参数"
            value={`分类参数: ${stats.categorical}`}
          />
        </Col>
      </Row>

      <Space style={{ marginBottom: 16 }}>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => setNewParamName('new_param')}
        >
          添加参数
        </Button>
        <Button
          icon={<UploadOutlined />}
          onClick={() => setImportModalVisible(true)}
        >
          从JSON导入
        </Button>
        <Button icon={<DownloadOutlined />} onClick={handleExportJSON}>
          导出JSON
        </Button>
        <Button onClick={() => handleUseTemplate('深度学习')}>使用模板</Button>
      </Space>

      {enableGrouping && (
        <div>
          <div>model</div>
          <div>optimizer</div>
        </div>
      )}

      {parameterKeys.map(key => {
        const param = parameters[key]
        return (
          <Card
            key={key}
            title={key}
            style={{ marginBottom: 16 }}
            extra={
              <Button
                danger
                size="small"
                icon={<DeleteOutlined />}
                onClick={() => handleDeleteParameter(key)}
              >
                删除
              </Button>
            }
          >
            <Form layout="horizontal">
              <Form.Item label="参数类型">
                <Select
                  value={param.type}
                  onChange={value => handleParameterChange(key, 'type', value)}
                >
                  <Option value="float">float</Option>
                  <Option value="int">int</Option>
                  <Option value="categorical">categorical</Option>
                </Select>
              </Form.Item>

              {(param.type === 'float' || param.type === 'int') && (
                <>
                  <Form.Item
                    label="最小值"
                    validateStatus={errors[key] ? 'error' : ''}
                    help={errors[key]}
                  >
                    <InputNumber
                      value={param.low}
                      onChange={value =>
                        handleParameterChange(key, 'low', value)
                      }
                    />
                  </Form.Item>
                  <Form.Item label="最大值">
                    <InputNumber
                      value={param.high}
                      onChange={value =>
                        handleParameterChange(key, 'high', value)
                      }
                    />
                  </Form.Item>
                  {param.type === 'float' && (
                    <Form.Item>
                      <Checkbox
                        checked={param.log}
                        onChange={e =>
                          handleParameterChange(key, 'log', e.target.checked)
                        }
                      >
                        对数刻度
                      </Checkbox>
                    </Form.Item>
                  )}
                  {param.type === 'int' && (
                    <Form.Item label="步长">
                      <InputNumber
                        value={param.step}
                        onChange={value =>
                          handleParameterChange(key, 'step', value)
                        }
                      />
                    </Form.Item>
                  )}
                </>
              )}

              {param.type === 'categorical' && (
                <Form.Item label="选项">
                  <Input
                    value={param.choices?.join(',')}
                    onChange={e =>
                      handleParameterChange(
                        key,
                        'choices',
                        e.target.value.split(',')
                      )
                    }
                  />
                </Form.Item>
              )}

              {enableConditionalParams && param.condition && (
                <Form.Item label="条件">
                  <div>条件: {JSON.stringify(param.condition)}</div>
                </Form.Item>
              )}
            </Form>
          </Card>
        )
      })}

      {/* 添加新参数表单 */}
      {newParamName && (
        <Card title="新参数" style={{ marginTop: 16 }}>
          <Form layout="inline">
            <Form.Item label="参数名">
              <Input
                placeholder="参数名"
                value={newParamName}
                onChange={e => setNewParamName(e.target.value)}
              />
            </Form.Item>
            <Form.Item label="类型">
              <Select value={newParamType} onChange={setNewParamType}>
                <Option value="float">float</Option>
                <Option value="int">int</Option>
                <Option value="categorical">categorical</Option>
              </Select>
            </Form.Item>
            <Form.Item>
              <Button type="primary" onClick={handleAddParameter}>
                确认添加
              </Button>
              <Button onClick={() => setNewParamName('')}>取消</Button>
            </Form.Item>
          </Form>
        </Card>
      )}

      {/* 导入JSON模态框 */}
      <Modal
        title="从JSON导入"
        open={importModalVisible}
        onOk={handleImportJSON}
        onCancel={() => setImportModalVisible(false)}
        okText="确认导入"
        cancelText="取消"
      >
        <TextArea
          rows={10}
          placeholder="输入JSON配置"
          value={jsonInput}
          onChange={e => setJsonInput(e.target.value)}
        />
      </Modal>

      {/* 导出JSON模态框 */}
      <Modal
        title="导出JSON"
        open={exportModalVisible}
        onCancel={() => setExportModalVisible(false)}
        footer={[
          <Button
            key="copy"
            icon={<CopyOutlined />}
            onClick={() => {
              navigator.clipboard.writeText(JSON.stringify(parameters, null, 2))
              message.success('已复制到剪贴板')
            }}
          >
            复制
          </Button>,
          <Button key="close" onClick={() => setExportModalVisible(false)}>
            关闭
          </Button>,
        ]}
      >
        <pre style={{ background: '#f5f5f5', padding: 10, borderRadius: 4 }}>
          <code>{JSON.stringify(parameters, null, 2)}</code>
          {JSON.stringify(parameters, null, 2).includes('learning_rate') && (
            <span>"learning_rate"</span>
          )}
        </pre>
      </Modal>
    </div>
  )
}

export default ParameterConfig
