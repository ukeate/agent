import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Table, Button, Alert, Space } from 'antd'

type Report = {
  report_id: string
  status: string
  generated_at?: string
  download_url?: string
}

const HyperparameterReportsPage: React.FC = () => {
  const [reports, setReports] = useState<Report[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await apiFetch(
        buildApiUrl('/api/v1/hyperparameter-optimization/experiments')
      )
      const data = await res.json()
      // 假定后端返回实验列表；报告下载走 /experiments/{id}/visualizations
      const mapped: Report[] = (data || []).map((exp: any) => ({
        report_id: exp.id,
        status: exp.status,
        generated_at: exp.completed_at,
        download_url: `/api/v1/hyperparameter-optimization/experiments/${exp.id}/visualizations`,
      }))
      setReports(mapped)
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setReports([])
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
        <Button onClick={load} loading={loading}>
          刷新
        </Button>
        {error && <Alert type="error" message={error} />}
        <Card title="超参实验报告">
          <Table
            rowKey="report_id"
            loading={loading}
            dataSource={reports}
            locale={{ emptyText: '暂无报告，请先创建并完成实验。' }}
            columns={[
              { title: '实验/报告ID', dataIndex: 'report_id' },
              { title: '状态', dataIndex: 'status' },
              { title: '完成时间', dataIndex: 'generated_at' },
              {
                title: '下载',
                render: (_, r) =>
                  r.download_url ? (
                    <a href={r.download_url} target="_blank" rel="noreferrer">
                      查看
                    </a>
                  ) : (
                    '-'
                  ),
              },
            ]}
          />
        </Card>
      </Space>
    </div>
  )
}

export default HyperparameterReportsPage
