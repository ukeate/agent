import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Row, Col, Statistic, Table, Tag, Button, Space, Alert, Typography } from 'antd'
import { ReloadOutlined, SafetyOutlined } from '@ant-design/icons'
import authService from '../services/authService'

type TrustMetric = { component: string; score: number; status: string; last_evaluated?: string; details?: string }
type RiskAssessment = { id: string; category: string; level: string; description?: string; mitigation_status?: string }
type SecurityIncident = { id: string; type: string; severity: string; status: string; description?: string; detection_time?: string }
type ComplianceCheck = { framework: string; requirement: string; status: string; last_check?: string }

const AiTrismPage: React.FC = () => {
  const [trust, setTrust] = useState<TrustMetric[]>([])
  const [risks, setRisks] = useState<RiskAssessment[]>([])
  const [incidents, setIncidents] = useState<SecurityIncident[]>([])
  const [compliance, setCompliance] = useState<ComplianceCheck[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const token = authService.getToken()
      const authHeaders = token ? { Authorization: `Bearer ${token}` } : undefined
      const [overviewRes, risksRes, auditRes, wlRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/model-service/monitoring/overview')),
        apiFetch(buildApiUrl('/api/v1/risk-assessment/health')),
        token ? apiFetch(buildApiUrl('/api/v1/security/mcp-tools/audit?limit=200'), { headers: authHeaders }) : Promise.resolve(null),
        token ? apiFetch(buildApiUrl('/api/v1/security/mcp-tools/whitelist'), { headers: authHeaders }) : Promise.resolve(null)
      ])
      const overview = await overviewRes.json()
      setTrust([
        {
          component: 'System',
          score: overview?.health_score ?? 0,
          status: (overview?.health_score ?? 0) > 80 ? 'high' : 'medium',
          last_evaluated: overview?.timestamp,
          details: '来自监控概览'
        }
      ])

      const risksJson = await risksRes.json()
      setRisks(Array.isArray(risksJson?.risks) ? risksJson.risks : [])

      const auditJson = auditRes ? await auditRes.json() : {}
      const incs: SecurityIncident[] = (auditJson?.logs || []).map((log: any, idx: number) => ({
        id: log.id || String(idx),
        type: log.event_type || 'audit',
        severity: log.result === 'failure' ? 'high' : 'low',
        status: 'detected',
        description: log.resource,
        detection_time: log.timestamp
      }))
      setIncidents(incs)

      const wlJson = wlRes ? await wlRes.json() : {}
      const whitelist = Array.isArray(wlJson?.whitelist) ? wlJson.whitelist : []
      const comps: ComplianceCheck[] = whitelist.map((name: string) => ({
        framework: 'MCP',
        requirement: `whitelist ${name}`,
        status: 'compliant',
        last_check: new Date().toISOString()
      }))
      setCompliance(comps)
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setTrust([])
      setRisks([])
      setIncidents([])
      setCompliance([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Space>
            <SafetyOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              AI Trism
            </Typography.Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && <Alert type="error" message={error} />}

        <Row gutter={16}>
          <Col span={6}>
            <Card>
              <Statistic title="信任分" value={trust[0]?.score ?? 0} />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic title="风险数量" value={risks.length} />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic title="安全事件" value={incidents.length} />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic title="合规检查" value={compliance.length} />
            </Card>
          </Col>
        </Row>

        <Card title="信任指标">
          <Table
            rowKey="component"
            dataSource={trust}
            loading={loading}
            columns={[
              { title: '组件', dataIndex: 'component' },
              { title: '评分', dataIndex: 'score' },
              {
                title: '状态',
                dataIndex: 'status',
                render: (s: string) => <Tag color={s === 'high' ? 'green' : 'orange'}>{s}</Tag>
              },
              { title: '时间', dataIndex: 'last_evaluated' }
            ]}
          />
        </Card>

        <Card title="风险评估">
          <Table
            rowKey="id"
            dataSource={risks}
            loading={loading}
            columns={[
              { title: 'ID', dataIndex: 'id' },
              { title: '类别', dataIndex: 'category' },
              { title: '等级', dataIndex: 'level' },
              { title: '缓解', dataIndex: 'mitigation_status' }
            ]}
          />
        </Card>

        <Card title="安全事件">
          <Table
            rowKey="id"
            dataSource={incidents}
            loading={loading}
            columns={[
              { title: 'ID', dataIndex: 'id' },
              { title: '类型', dataIndex: 'type' },
              { title: '严重性', dataIndex: 'severity' },
              { title: '状态', dataIndex: 'status' },
              { title: '时间', dataIndex: 'detection_time' }
            ]}
          />
        </Card>

        <Card title="合规检查">
          <Table
            rowKey="requirement"
            dataSource={compliance}
            loading={loading}
            columns={[
              { title: '框架', dataIndex: 'framework' },
              { title: '要求', dataIndex: 'requirement' },
              { title: '状态', dataIndex: 'status' },
              { title: '时间', dataIndex: 'last_check' }
            ]}
          />
        </Card>
      </Space>
    </div>
  )
}

export default AiTrismPage
