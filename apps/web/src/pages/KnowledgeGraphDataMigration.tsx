import React, { useEffect, useState } from 'react'
import { Card, Table, Tag, Space, Button, Typography, message } from 'antd'
import {
  SwapOutlined,
  ReloadOutlined,
  PlayCircleOutlined,
  HistoryOutlined,
  RollbackOutlined,
} from '@ant-design/icons'
import {
  knowledgeGraphService,
  type Migration,
  type MigrationRecord,
} from '../services/knowledgeGraphService'

const { Title, Text } = Typography

const statusColor: Record<string, string> = {
  pending: 'orange',
  running: 'blue',
  completed: 'green',
  failed: 'red',
  rolled_back: 'purple',
}

const KnowledgeGraphDataMigration: React.FC = () => {
  const [migrations, setMigrations] = useState<Migration[]>([])
  const [records, setRecords] = useState<MigrationRecord[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    try {
      const [migs, recs] = await Promise.all([
        knowledgeGraphService.listMigrations(),
        knowledgeGraphService.getMigrationRecords(),
      ])
      setMigrations(migs)
      setRecords(recs)
    } catch (error: any) {
      message.error(error?.message || '加载迁移数据失败')
    } finally {
      setLoading(false)
    }
  }

  const runMigration = async (id: string) => {
    setLoading(true)
    try {
      await knowledgeGraphService.applyMigration(id)
      message.success('迁移执行完成')
      await loadData()
    } catch (error: any) {
      message.error(error?.message || '迁移执行失败')
    } finally {
      setLoading(false)
    }
  }

  const runAll = async () => {
    setLoading(true)
    try {
      await knowledgeGraphService.applyAllMigrations()
      message.success('所有待迁移已执行')
      await loadData()
    } catch (error: any) {
      message.error(error?.message || '执行失败')
    } finally {
      setLoading(false)
    }
  }

  const rollback = async (id: string) => {
    setLoading(true)
    try {
      await knowledgeGraphService.rollbackMigration(id)
      message.success('迁移已回滚')
      await loadData()
    } catch (error: any) {
      message.error(error?.message || '回滚失败')
    } finally {
      setLoading(false)
    }
  }

  const migrationColumns = [
    {
      title: '迁移ID',
      dataIndex: 'id',
      key: 'id',
      width: 160,
    },
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '类型',
      dataIndex: 'migration_type',
      key: 'migration_type',
      render: (type: string) => <Tag>{type}</Tag>,
      width: 140,
    },
    {
      title: '版本',
      dataIndex: 'version',
      key: 'version',
      width: 100,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (status: string) => (
        <Tag color={statusColor[status] || 'default'}>{status}</Tag>
      ),
    },
    {
      title: '依赖',
      dataIndex: 'dependencies',
      key: 'dependencies',
      render: (deps: string[]) =>
        deps && deps.length ? (
          deps.map(dep => <Tag key={dep}>{dep}</Tag>)
        ) : (
          <Text type="secondary">无</Text>
        ),
      width: 200,
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time: string) => new Date(time).toLocaleString(),
      width: 200,
    },
    {
      title: '操作',
      key: 'actions',
      width: 200,
      render: (_: any, record: Migration) => (
        <Space>
          <Button
            type="primary"
            icon={<PlayCircleOutlined />}
            onClick={() => runMigration(record.id)}
            disabled={loading}
          >
            执行
          </Button>
          <Button
            icon={<RollbackOutlined />}
            onClick={() => rollback(record.id)}
            disabled={loading || record.status !== 'completed'}
          >
            回滚
          </Button>
        </Space>
      ),
    },
  ]

  const recordColumns = [
    {
      title: '迁移ID',
      dataIndex: 'migration_id',
      key: 'migration_id',
      width: 160,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (status: string) => (
        <Tag color={statusColor[status] || 'default'}>{status}</Tag>
      ),
    },
    {
      title: '开始时间',
      dataIndex: 'started_at',
      key: 'started_at',
      render: (time: string) => new Date(time).toLocaleString(),
      width: 180,
    },
    {
      title: '完成时间',
      dataIndex: 'completed_at',
      key: 'completed_at',
      render: (time?: string) =>
        time ? new Date(time).toLocaleString() : '进行中',
      width: 180,
    },
    {
      title: '耗时(ms)',
      dataIndex: 'execution_time_ms',
      key: 'execution_time_ms',
      width: 120,
    },
    {
      title: '影响节点',
      dataIndex: 'affected_nodes',
      key: 'affected_nodes',
      width: 120,
    },
    {
      title: '影响关系',
      dataIndex: 'affected_relationships',
      key: 'affected_relationships',
      width: 120,
    },
    {
      title: '错误信息',
      dataIndex: 'error_message',
      key: 'error_message',
      render: (msg?: string) => msg || '-',
    },
  ]

  return (
    <div style={{ padding: 24 }}>
      <Space style={{ marginBottom: 16 }}>
        <SwapOutlined />
        <Title level={3} style={{ margin: 0 }}>
          知识图谱数据迁移
        </Title>
      </Space>

      <Card
        title="迁移任务"
        extra={
          <Space>
            <Button
              icon={<ReloadOutlined />}
              onClick={loadData}
              loading={loading}
            >
              刷新
            </Button>
            <Button type="primary" onClick={runAll} loading={loading}>
              执行全部待迁移
            </Button>
          </Space>
        }
      >
        <Table
          rowKey="id"
          loading={loading}
          columns={migrationColumns}
          dataSource={migrations}
          pagination={{ pageSize: 8 }}
        />
      </Card>

      <Card
        style={{ marginTop: 16 }}
        title="迁移记录"
        extra={<HistoryOutlined />}
      >
        <Table
          rowKey={record => `${record.migration_id}-${record.started_at}`}
          loading={loading}
          columns={recordColumns}
          dataSource={records}
          pagination={{ pageSize: 10 }}
        />
      </Card>
    </div>
  )
}

export default KnowledgeGraphDataMigration
