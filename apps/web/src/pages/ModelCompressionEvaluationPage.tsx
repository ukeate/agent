import React, { useState } from 'react'
import { Button, Card, Form, Input, Space, Typography, message } from 'antd'
import apiClient from '../services/apiClient'

const { Title, Paragraph } = Typography

const ModelCompressionEvaluationPage: React.FC = () => {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [report, setReport] = useState<any>(null)

  const loadResult = async (jobId: string) => {
    const resp = await apiClient.get(`/model-compression/results/${jobId}`)
    return resp.data
  }

  const loadReport = async (jobId: string) => {
    const resp = await apiClient.get(`/model-compression/results/${jobId}/report`)
    return resp.data
  }

  const onLoad = async (values: any) => {
    setLoading(true)
    try {
      const jobId = String(values.job_id || '').trim()
      if (!jobId) {
        message.warning('请输入 job_id')
        return
      }
      const [r, rep] = await Promise.allSettled([loadResult(jobId), loadReport(jobId)])
      setResult(r.status === 'fulfilled' ? r.value : null)
      setReport(rep.status === 'fulfilled' ? rep.value : null)
      message.success('加载完成')
    } catch (e: any) {
      message.error(e?.message || '加载失败')
      setResult(null)
      setReport(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: 24 }}>
      <Title level={2} style={{ margin: 0 }}>压缩评估与对比</Title>
      <Paragraph style={{ marginTop: 8 }}>
        通过 job_id 查询压缩结果与评估报告：`/api/v1/model-compression/results/{'{job_id}'}` 与 `/api/v1/model-compression/results/{'{job_id}'}/report`。
      </Paragraph>

      <Card title="查询" style={{ marginBottom: 16 }}>
        <Form form={form} layout="inline" onFinish={onLoad}>
          <Form.Item name="job_id" rules={[{ required: true, message: '请输入 job_id' }]}>
            <Input placeholder="job_id" style={{ width: 320 }} />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>加载</Button>
              <Button onClick={() => { setResult(null); setReport(null) }} disabled={loading}>清空</Button>
            </Space>
          </Form.Item>
        </Form>
      </Card>

      <Card title="压缩结果" style={{ marginBottom: 16 }}>
        <pre style={{ background: '#f5f5f5', padding: 12, borderRadius: 4, minHeight: 120 }}>
          {result ? JSON.stringify(result, null, 2) : '暂无数据'}
        </pre>
      </Card>

      <Card title="评估报告">
        <pre style={{ background: '#f5f5f5', padding: 12, borderRadius: 4, minHeight: 120 }}>
          {report ? JSON.stringify(report, null, 2) : '暂无数据'}
        </pre>
      </Card>
    </div>
  )
}

export default ModelCompressionEvaluationPage
