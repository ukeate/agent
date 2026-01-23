import React, { useEffect, useMemo, useState } from 'react'
import {
  Card,
  Row,
  Col,
  Typography,
  Progress,
  Statistic,
  Select,
  Space,
  Button,
  Tag,
  Tabs,
  message,
} from 'antd'
import {
  MonitorOutlined,
  ReloadOutlined,
  FileTextOutlined,
  LineChartOutlined,
} from '@ant-design/icons'
import { fineTuningService, TrainingJob } from '../services/fineTuningService'

import { logger } from '../utils/logger'
const { Title, Text } = Typography

const FineTuningMonitorPage: React.FC = () => {
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [jobId, setJobId] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [progressInfo, setProgressInfo] = useState<any>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [metrics, setMetrics] = useState<any>(null)

  const selectedJob = useMemo(
    () => jobs.find(j => j.job_id === jobId) || null,
    [jobs, jobId]
  )

  const loadJobs = async () => {
    const list = await fineTuningService.getTrainingJobs()
    setJobs(list)
    if (!jobId && list[0]?.job_id) setJobId(list[0].job_id)
  }

  const loadDetails = async (id: string) => {
    const [p, l, m] = await Promise.all([
      fineTuningService.getTrainingProgress(id),
      fineTuningService.getTrainingLogs(id, 200),
      fineTuningService.getTrainingMetrics(id),
    ])
    setProgressInfo(p)
    setLogs(l.logs || [])
    setMetrics(m.metrics || null)
  }

  const refresh = async () => {
    try {
      setLoading(true)
      await loadJobs()
      if (jobId) await loadDetails(jobId)
    } catch (e) {
      logger.error('刷新监控失败:', e)
      message.error('刷新监控失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refresh()
  }, [])

  useEffect(() => {
    if (!jobId) return
    loadDetails(jobId).catch(e => logger.error('加载监控详情失败:', e))
  }, [jobId])

  const statusTag = (status?: string) => {
    const map: Record<string, { color: string; text: string }> = {
      running: { color: 'processing', text: '运行中' },
      completed: { color: 'success', text: '已完成' },
      failed: { color: 'error', text: '失败' },
      pending: { color: 'default', text: '等待中' },
      paused: { color: 'warning', text: '已暂停' },
      cancelled: { color: 'default', text: '已取消' },
    }
    const v = status ? map[status] : null
    return (
      <Tag color={v?.color || 'default'}>{v?.text || status || '未知'}</Tag>
    )
  }

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <MonitorOutlined style={{ marginRight: 8, color: '#52c41a' }} />
          训练监控中心
        </Title>
        <Text type="secondary">
          基于真实训练任务输出（progress/logs/metrics）进行监控
        </Text>
      </div>

      <Card style={{ marginBottom: 16 }}>
        <Space wrap>
          <Select
            style={{ width: 520 }}
            placeholder="选择任务"
            value={jobId || undefined}
            onChange={setJobId}
            options={jobs.map(j => ({
              value: j.job_id,
              label: `${j.job_name} (${j.job_id})`,
            }))}
          />
          <Button icon={<ReloadOutlined />} onClick={refresh} loading={loading}>
            刷新
          </Button>
          {selectedJob ? statusTag(selectedJob.status) : null}
        </Space>
      </Card>

      {selectedJob ? (
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={6}>
            <Card>
              <Statistic
                title="状态"
                value={selectedJob.status}
                prefix={<LineChartOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic title="进度" value={selectedJob.progress} suffix="%" />
              <Progress
                percent={selectedJob.progress}
                size="small"
                style={{ marginTop: 8 }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Epoch"
                value={`${selectedJob.current_epoch}/${selectedJob.total_epochs}`}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="耗时(秒)"
                value={progressInfo?.elapsed_time ?? '-'}
              />
            </Card>
          </Col>
        </Row>
      ) : null}

      <Card>
        <Tabs
          items={[
            {
              key: 'logs',
              label: '日志',
              children: (
                <div
                  style={{
                    maxHeight: 360,
                    overflow: 'auto',
                    background: '#f5f5f5',
                    padding: 12,
                    fontFamily: 'monospace',
                    fontSize: 12,
                  }}
                >
                  {logs.length ? (
                    logs.map((line, i) => <div key={i}>{line}</div>)
                  ) : (
                    <Text type="secondary">暂无日志</Text>
                  )}
                </div>
              ),
            },
            {
              key: 'metrics',
              label: '指标报告',
              children: (
                <pre
                  style={{
                    maxHeight: 360,
                    overflow: 'auto',
                    background: '#f5f5f5',
                    padding: 12,
                    fontSize: 12,
                  }}
                >
                  {metrics ? JSON.stringify(metrics, null, 2) : '{}'}
                </pre>
              ),
            },
            {
              key: 'progress',
              label: '进度详情',
              children: (
                <pre
                  style={{
                    maxHeight: 360,
                    overflow: 'auto',
                    background: '#f5f5f5',
                    padding: 12,
                    fontSize: 12,
                  }}
                >
                  {progressInfo ? JSON.stringify(progressInfo, null, 2) : '{}'}
                </pre>
              ),
            },
          ]}
        />
      </Card>

      {selectedJob?.error_message ? (
        <Card title="错误信息" style={{ marginTop: 16 }}>
          <Text type="danger">
            <FileTextOutlined style={{ marginRight: 8 }} />
            {selectedJob.error_message}
          </Text>
        </Card>
      ) : null}
    </div>
  )
}

export default FineTuningMonitorPage
