import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Table, Button, Space, Typography, Tag, Alert, Modal, Form, Input, Select, message } from 'antd'
import { ReloadOutlined, AlertOutlined, PhoneOutlined } from '@ant-design/icons'

type CrisisAssessment = {
  user_id: string
  severity_level?: string
  crisis_type?: string
  risk_score?: number
  timestamp?: string
}

type EmergencyContact = {
  id: string
  name: string
  type: string
  phone: string
  availability?: string
}

type SupportResource = {
  id: string
  title: string
  type: string
  description?: string
}

const { Option } = Select

const CrisisDetectionSupportPage: React.FC = () => {
  const [assessments, setAssessments] = useState<CrisisAssessment[]>([])
  const [contacts, setContacts] = useState<EmergencyContact[]>([])
  const [resources, setResources] = useState<SupportResource[]>([])
  const [loading, setLoading] = useState(false)
  const [detectVisible, setDetectVisible] = useState(false)
  const [form] = Form.useForm()

  const load = async () => {
    setLoading(true)
    try {
      const [aRes, cRes, rRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/social-emotion/crisis/assessments')),
        apiFetch(buildApiUrl('/api/v1/social-emotion/crisis/contacts')),
        apiFetch(buildApiUrl('/api/v1/social-emotion/crisis/resources'))
      ])
      setAssessments((await aRes.json())?.assessments || [])
      setContacts((await cRes.json())?.contacts || [])
      setResources((await rRes.json())?.resources || [])
    } catch (e: any) {
      message.error(e?.message || '加载失败')
      setAssessments([])
      setContacts([])
      setResources([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const runDetection = async (values: any) => {
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/social-emotion/crisis/detect'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values)
      })
      message.success('检测请求已提交')
      setDetectVisible(false)
      form.resetFields()
      load()
    } catch (e: any) {
      message.error(e?.message || '提交失败')
    }
  }

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Space>
            <AlertOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              危机检测与支持
            </Typography.Title>
          </Space>
          <Space>
            <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
              刷新
            </Button>
            <Button type="primary" onClick={() => setDetectVisible(true)}>
              提交检测
            </Button>
          </Space>
        </Space>

        <Card title="危机评估">
          <Table
            rowKey="user_id"
            dataSource={assessments}
            loading={loading}
            columns={[
              { title: '用户', dataIndex: 'user_id' },
              { title: '类型', dataIndex: 'crisis_type' },
              { title: '严重程度', dataIndex: 'severity_level', render: (s) => <Tag color="red">{s}</Tag> },
              { title: '风险', dataIndex: 'risk_score' },
              { title: '时间', dataIndex: 'timestamp' }
            ]}
            locale={{ emptyText: '暂无危机评估' }}
          />
        </Card>

        <Card title="紧急联系人">
          <Table
            rowKey="id"
            dataSource={contacts}
            loading={loading}
            columns={[
              { title: '名称', dataIndex: 'name' },
              { title: '类型', dataIndex: 'type' },
              { title: '电话', dataIndex: 'phone' },
              { title: '可用性', dataIndex: 'availability' }
            ]}
            locale={{ emptyText: '暂无联系人' }}
          />
        </Card>

        <Card title="支持资源">
          <Table
            rowKey="id"
            dataSource={resources}
            loading={loading}
            columns={[
              { title: '标题', dataIndex: 'title' },
              { title: '类型', dataIndex: 'type' },
              { title: '描述', dataIndex: 'description' }
            ]}
            locale={{ emptyText: '暂无资源' }}
          />
        </Card>
      </Space>

      <Modal
        title="提交危机检测"
        open={detectVisible}
        onCancel={() => setDetectVisible(false)}
        onOk={() => form.submit()}
        confirmLoading={loading}
      >
        <Form layout="vertical" form={form} onFinish={runDetection}>
          <Form.Item name="user_id" label="用户ID" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
          <Form.Item name="context" label="上下文">
            <TextArea rows={3} />
          </Form.Item>
          <Form.Item name="crisis_type" label="类型">
            <Select allowClear>
              <Option value="suicidal_ideation">自杀意念</Option>
              <Option value="anxiety_attack">焦虑发作</Option>
              <Option value="panic">惊恐</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default CrisisDetectionSupportPage
