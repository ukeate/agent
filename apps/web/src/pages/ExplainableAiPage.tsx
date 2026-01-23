import React, { useEffect, useState } from 'react'
import { Alert, Button, Card, Space, Tabs, Typography } from 'antd'
import explainableAiService, {
  type DecisionExplanation,
  type DemoScenarios,
  type ExplanationTypes,
} from '../services/explainableAiService'

import { logger } from '../utils/logger'
const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs

type CustomFactor = {
  factor_name: string
  factor_value: string
  weight: string
  impact: string
}

const ExplainableAiPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [exportStatus, setExportStatus] = useState<string>('')

  const [explanation, setExplanation] = useState<DecisionExplanation | null>(
    null
  )
  const [explanationTypes, setExplanationTypes] =
    useState<ExplanationTypes | null>(null)
  const [demoScenarios, setDemoScenarios] = useState<DemoScenarios | null>(null)

  const [activeTab, setActiveTab] = useState<'demo' | 'custom'>('demo')

  const [selectedExplanationType, setSelectedExplanationType] =
    useState('decision')
  const [selectedExplanationLevel, setSelectedExplanationLevel] =
    useState('detailed')

  const [selectedScenario, setSelectedScenario] = useState('loan_approval')
  const [selectedComplexity, setSelectedComplexity] = useState('medium')
  const [includeCot, setIncludeCot] = useState(false)

  const [customDecisionId, setCustomDecisionId] = useState('')
  const [customDecisionContext, setCustomDecisionContext] = useState('')
  const [customFactors, setCustomFactors] = useState<CustomFactor[]>([])

  // 加载配置数据
  const loadConfigurations = async (): Promise<void> => {
    try {
      const [typesData, scenariosData] = await Promise.all([
        explainableAiService.getExplanationTypes(),
        explainableAiService.getDemoScenarios(),
      ])
      setExplanationTypes(typesData)
      setDemoScenarios(scenariosData)
    } catch (error) {
      logger.error('加载配置失败:', error)
      setError('配置加载失败')
    }
  }

  const addFactorRow = (): void => {
    setCustomFactors(prev => [
      ...prev,
      { factor_name: '', factor_value: '', weight: '', impact: '' },
    ])
  }

  const generateDemoExplanation = async (): Promise<void> => {
    setError(null)
    setLoading(true)
    try {
      const result = await explainableAiService.generateDemoScenario({
        scenario_type: selectedScenario as any,
        complexity: selectedComplexity as any,
        include_cot: includeCot,
      })
      setExplanation(result)
    } catch (error) {
      logger.error('生成示例解释失败:', error)
      setError('解释生成失败')
    } finally {
      setLoading(false)
    }
  }

  const generateCustomExplanation = async (): Promise<void> => {
    setError(null)
    if (!customDecisionId.trim() || !customDecisionContext.trim()) {
      setError('请填写决策ID和决策上下文')
      return
    }
    const factors = customFactors.map(f => {
      const weight = Number(f.weight)
      const impact = Number(f.impact)
      if (
        !f.factor_name.trim() ||
        !f.factor_value.trim() ||
        !Number.isFinite(weight) ||
        !Number.isFinite(impact)
      ) {
        throw new Error('invalid_factor')
      }
      return {
        factor_name: f.factor_name.trim(),
        factor_value: f.factor_value.trim(),
        weight,
        impact,
        source: 'user',
      }
    })

    setLoading(true)
    try {
      const result = await explainableAiService.generateExplanation({
        decision_id: customDecisionId.trim(),
        decision_context: customDecisionContext.trim(),
        explanation_type: selectedExplanationType as any,
        explanation_level: selectedExplanationLevel as any,
        factors,
      })
      setExplanation(result)
    } catch (error) {
      logger.error('生成自定义解释失败:', error)
      setError('解释生成失败')
    } finally {
      setLoading(false)
    }
  }

  const generateByActiveTab = async (): Promise<void> => {
    try {
      if (activeTab === 'custom') {
        await generateCustomExplanation()
        return
      }
      await generateDemoExplanation()
    } catch {
      setError('解释生成失败')
    }
  }

  const exportExplanation = (format: string): void => {
    if (!explanation) return
    setExportStatus('正在导出')
    const percent = Math.round(
      (explanation.confidence_metrics?.overall_confidence || 0) * 100
    )
    const content =
      format === 'json'
        ? JSON.stringify(explanation, null, 2)
        : format === 'markdown'
          ? `# Explanation\n\n- decision_id: ${explanation.decision_id}\n- outcome: ${explanation.decision_outcome}\n\n## Summary\n\n${explanation.summary_explanation}\n`
          : format === 'html'
            ? `<h1>Explanation</h1><div><b>decision_id</b>: ${explanation.decision_id}</div><div><b>outcome</b>: ${explanation.decision_outcome}</div><div><b>confidence</b>: ${percent}%</div><h2>Summary</h2><pre>${explanation.summary_explanation}</pre>`
            : format === 'xml'
              ? `<explanation><decision_id>${explanation.decision_id}</decision_id><outcome>${explanation.decision_outcome}</outcome><summary>${explanation.summary_explanation}</summary></explanation>`
              : `decision_id: ${explanation.decision_id}\noutcome: ${explanation.decision_outcome}\nconfidence: ${percent}%\n\n${explanation.summary_explanation}`

    const ext = format === '纯文本' ? 'txt' : format
    const blob = new Blob([content], {
      type: format === 'html' ? 'text/html' : 'text/plain',
    })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `explanation.${ext}`
    document.body.appendChild(a)
    a.click()
    window.URL.revokeObjectURL(url)
    document.body.removeChild(a)
  }

  useEffect(() => {
    loadConfigurations()
  }, [])

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        {/* 页面标题 */}
        <div style={{ marginBottom: '24px' }}>
          <Title level={1}>可解释AI决策系统</Title>
          <Paragraph>
            通过透明化的决策过程分析，帮助理解AI系统的推理逻辑和决策依据。
          </Paragraph>
        </div>

        {/* 控制面板 */}
        <Card
          style={{ marginBottom: '24px' }}
          data-testid="explanation-controls"
        >
          <div style={{ padding: '16px' }}>
            {!explanationTypes || !demoScenarios ? (
              <div>加载配置中...</div>
            ) : (
              <>
                <Space size="large" wrap>
                  <label>
                    <Text>解释类型:</Text>{' '}
                    <select
                      data-testid="explanation-type-select"
                      value={selectedExplanationType}
                      onChange={e => setSelectedExplanationType(e.target.value)}
                    >
                      {explanationTypes.explanation_types.map(t => (
                        <option key={t.value} value={t.value}>
                          {t.value}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    <Text>解释级别:</Text>{' '}
                    <select
                      data-testid="explanation-level-select"
                      value={selectedExplanationLevel}
                      onChange={e =>
                        setSelectedExplanationLevel(e.target.value)
                      }
                    >
                      {explanationTypes.explanation_levels.map(l => (
                        <option key={l.value} value={l.value}>
                          {l.value}
                        </option>
                      ))}
                    </select>
                  </label>
                  <Button
                    type="primary"
                    data-testid="generate-explanation-btn"
                    onClick={generateByActiveTab}
                    disabled={loading}
                  >
                    生成解释
                  </Button>
                </Space>

                <div style={{ marginTop: '16px' }}>
                  <Tabs
                    data-testid="tab-container"
                    activeKey={activeTab}
                    onChange={key => setActiveTab(key as any)}
                  >
                    <TabPane
                      tab={
                        <span data-testid="demo-scenarios-tab">演示场景</span>
                      }
                      key="demo"
                    >
                      <Space size="large" wrap>
                        <label>
                          <Text>场景:</Text>{' '}
                          <select
                            data-testid="scenario-select"
                            value={selectedScenario}
                            onChange={e => setSelectedScenario(e.target.value)}
                          >
                            {demoScenarios.scenarios.map(s => (
                              <option key={s.type} value={s.type}>
                                {s.type}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label>
                          <Text>复杂度:</Text>{' '}
                          <select
                            data-testid="complexity-select"
                            value={selectedComplexity}
                            onChange={e =>
                              setSelectedComplexity(e.target.value)
                            }
                          >
                            <option value="simple">simple</option>
                            <option value="medium">medium</option>
                            <option value="complex">complex</option>
                          </select>
                        </label>
                        <label>
                          <Text>CoT推理:</Text>{' '}
                          <input
                            data-testid="cot-reasoning-checkbox"
                            type="checkbox"
                            checked={includeCot}
                            onChange={e => setIncludeCot(e.target.checked)}
                          />
                        </label>
                        <Button
                          data-testid="generate-demo-btn"
                          onClick={generateDemoExplanation}
                          disabled={loading}
                        >
                          生成演示解释
                        </Button>
                      </Space>
                    </TabPane>

                    <TabPane
                      tab={
                        <span data-testid="custom-explanation-tab">
                          自定义解释
                        </span>
                      }
                      key="custom"
                    >
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <label>
                          <Text>决策ID:</Text>{' '}
                          <input
                            data-testid="decision-id-input"
                            value={customDecisionId}
                            onChange={e => setCustomDecisionId(e.target.value)}
                          />
                        </label>
                        <label>
                          <Text>决策上下文:</Text>{' '}
                          <textarea
                            data-testid="decision-context-input"
                            value={customDecisionContext}
                            onChange={e =>
                              setCustomDecisionContext(e.target.value)
                            }
                          />
                        </label>

                        <Button
                          data-testid="add-factor-btn"
                          onClick={addFactorRow}
                          disabled={loading}
                        >
                          添加因子
                        </Button>

                        {customFactors.map((f, idx) => (
                          <Space key={idx} wrap>
                            <input
                              data-testid="factor-name-input"
                              placeholder="factor_name"
                              value={f.factor_name}
                              onChange={e => {
                                const v = e.target.value
                                setCustomFactors(prev =>
                                  prev.map((x, i) =>
                                    i === idx ? { ...x, factor_name: v } : x
                                  )
                                )
                              }}
                            />
                            <input
                              data-testid="factor-value-input"
                              placeholder="factor_value"
                              value={f.factor_value}
                              onChange={e => {
                                const v = e.target.value
                                setCustomFactors(prev =>
                                  prev.map((x, i) =>
                                    i === idx ? { ...x, factor_value: v } : x
                                  )
                                )
                              }}
                            />
                            <input
                              data-testid="factor-weight-input"
                              placeholder="weight"
                              value={f.weight}
                              onChange={e => {
                                const v = e.target.value
                                setCustomFactors(prev =>
                                  prev.map((x, i) =>
                                    i === idx ? { ...x, weight: v } : x
                                  )
                                )
                              }}
                            />
                            <input
                              data-testid="factor-impact-input"
                              placeholder="impact"
                              value={f.impact}
                              onChange={e => {
                                const v = e.target.value
                                setCustomFactors(prev =>
                                  prev.map((x, i) =>
                                    i === idx ? { ...x, impact: v } : x
                                  )
                                )
                              }}
                            />
                          </Space>
                        ))}

                        <Button
                          type="primary"
                          data-testid="generate-custom-btn"
                          onClick={generateCustomExplanation}
                          disabled={loading}
                        >
                          生成自定义解释
                        </Button>
                      </Space>
                    </TabPane>
                  </Tabs>
                </div>
              </>
            )}
          </div>
        </Card>

        {/* 解释结果 */}
        <Card data-testid="explanation-result">
          <div style={{ padding: '16px' }}>
            {loading && <div data-testid="loading-indicator">加载中...</div>}
            {error && (
              <Alert
                data-testid="error-message"
                type="error"
                message={error}
                showIcon
                style={{ marginBottom: '16px' }}
              />
            )}

            {explanation ? (
              <>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div data-testid="decision-id">
                    决策ID: {explanation.decision_id}
                  </div>
                  <div data-testid="decision-outcome">
                    决策结果: {explanation.decision_outcome}
                  </div>
                  <div data-testid="summary-explanation">
                    {explanation.summary_explanation}
                  </div>
                  <div data-testid="confidence-metrics">
                    整体置信度:{' '}
                    {Math.round(
                      (explanation.confidence_metrics?.overall_confidence ||
                        0) * 100
                    )}
                    %
                  </div>
                </Space>

                <div data-testid="export-buttons" style={{ marginTop: '16px' }}>
                  <Space wrap>
                    <Button
                      data-testid="export-html-btn"
                      onClick={() => exportExplanation('html')}
                    >
                      HTML
                    </Button>
                    <Button
                      data-testid="export-markdown-btn"
                      onClick={() => exportExplanation('markdown')}
                    >
                      Markdown
                    </Button>
                    <Button
                      data-testid="export-json-btn"
                      onClick={() => exportExplanation('json')}
                    >
                      JSON
                    </Button>
                    <Button
                      data-testid="export-xml-btn"
                      onClick={() => exportExplanation('xml')}
                    >
                      XML
                    </Button>
                    <Button
                      data-testid="export-纯文本-btn"
                      onClick={() => exportExplanation('纯文本')}
                    >
                      纯文本
                    </Button>
                    <span data-testid="export-status">{exportStatus}</span>
                  </Space>
                </div>

                <Tabs
                  defaultActiveKey="components"
                  style={{ marginTop: '16px' }}
                >
                  <TabPane
                    tab={<span data-testid="components-tab">解释组件</span>}
                    key="components"
                  >
                    <table data-testid="factors-table">
                      <thead>
                        <tr>
                          <th>factor_name</th>
                          <th>factor_value</th>
                          <th>weight</th>
                          <th>impact</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(explanation.components || []).map(
                          (c: any, i: number) => (
                            <tr key={i}>
                              <td>{c.factor_name}</td>
                              <td>{String(c.factor_value)}</td>
                              <td>{Math.round((c.weight || 0) * 100)}%</td>
                              <td>
                                {Math.round((c.impact_score || 0) * 100)}%
                              </td>
                            </tr>
                          )
                        )}
                      </tbody>
                    </table>
                  </TabPane>

                  <TabPane
                    tab={
                      <span data-testid="counterfactuals-tab">反事实分析</span>
                    }
                    key="counterfactuals"
                  >
                    <Space direction="vertical" style={{ width: '100%' }}>
                      {(explanation.counterfactuals || []).map(
                        (cf: any, i: number) => (
                          <Card key={i} data-testid="counterfactual-card">
                            <div>{cf.scenario_name}</div>
                            <div>
                              {JSON.stringify(cf.changed_factors || {})}
                            </div>
                            <div>{cf.predicted_outcome}</div>
                            <div>
                              {Math.round((cf.probability || 0) * 100)}%
                            </div>
                            <div>{cf.explanation}</div>
                          </Card>
                        )
                      )}
                    </Space>
                  </TabPane>

                  <TabPane
                    tab={<span data-testid="visualization-tab">可视化</span>}
                    key="visualization"
                  >
                    <div data-testid="chart-toggle-buttons">
                      <button type="button">柱状图</button>
                      <button type="button">饼图</button>
                    </div>
                    <div
                      data-testid="factor-importance-chart"
                      style={{ marginTop: '12px' }}
                    >
                      {JSON.stringify(
                        (explanation as any).visualization_data
                          ?.factor_importance || {}
                      )}
                    </div>
                    <div
                      data-testid="confidence-breakdown-chart"
                      style={{ marginTop: '12px' }}
                    >
                      {JSON.stringify(
                        (explanation as any).visualization_data
                          ?.confidence_breakdown || {}
                      )}
                    </div>
                  </TabPane>
                </Tabs>
              </>
            ) : (
              <Alert message="暂无解释数据" type="info" showIcon />
            )}
          </div>
        </Card>
      </div>
    </div>
  )
}

export default ExplainableAiPage
