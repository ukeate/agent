/**
 * 实验列表组件
 */
import React, { useState, useEffect } from 'react';
import { Table, Button, Tag, Space, Input, Select, Modal, message, Spin, Empty, Checkbox } from 'antd';
import { PlusOutlined, ReloadOutlined, DeleteOutlined, PlayCircleOutlined, PauseOutlined, SearchOutlined } from '@ant-design/icons';
import apiClient from '../../services/apiClient';

import { logger } from '../../utils/logger'
const { Search } = Input;
const { Option } = Select;

interface Experiment {
  id: number;
  name: string;
  description: string;
  state: string;
  algorithm: string;
  created_at: string;
  parameter_ranges: any;
  optimization_config?: any;
}

const ExperimentList: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedRows, setSelectedRows] = useState<number[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [stateFilter, setStateFilter] = useState<string | undefined>();
  const [deleteModalVisible, setDeleteModalVisible] = useState(false);
  const [selectedExperiment, setSelectedExperiment] = useState<Experiment | null>(null);

  const fetchExperiments = async () => {
    setLoading(true);
    try {
      let url = '/api/v1/hyperparameter-optimization/experiments';
      if (searchQuery) {
        url = `/api/v1/hyperparameter-optimization/experiments/search?q=${searchQuery}`;
      }
      const response = await apiClient.get(url);
      let data = response.data || [];
      
      // 应用状态筛选
      if (stateFilter) {
        data = data.filter((exp: Experiment) => exp.state === stateFilter);
      }
      
      setExperiments(data);
    } catch (error) {
      message.error('加载实验失败');
      logger.error('加载实验失败:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchExperiments();
  }, [searchQuery, stateFilter]);

  const handleStart = async (id: number) => {
    try {
      await apiClient.post(`/hyperparameter-optimization/experiments/${id}/start`);
      message.success('实验已启动');
      fetchExperiments();
    } catch (error) {
      message.error('启动实验失败');
    }
  };

  const handleStop = async (id: number) => {
    try {
      await apiClient.post(`/hyperparameter-optimization/experiments/${id}/stop`);
      message.success('实验已停止');
      fetchExperiments();
    } catch (error) {
      message.error('停止实验失败');
    }
  };

  const handleDelete = async (id?: number) => {
    try {
      if (id) {
        await apiClient.delete(`/hyperparameter-optimization/experiments/${id}`);
      } else {
        // 批量删除
        for (const expId of selectedRows) {
          await apiClient.delete(`/hyperparameter-optimization/experiments/${expId}`);
        }
      }
      message.success('删除成功');
      setDeleteModalVisible(false);
      setSelectedExperiment(null);
      setSelectedRows([]);
      fetchExperiments();
    } catch (error) {
      message.error('删除失败');
    }
  };

  const getStateTag = (state: string) => {
    const stateMap: { [key: string]: { color: string; text: string } } = {
      created: { color: 'default', text: '已创建' },
      running: { color: 'processing', text: '运行中' },
      completed: { color: 'success', text: '已完成' },
      failed: { color: 'error', text: '失败' },
      stopped: { color: 'warning', text: '已停止' }
    };
    const config = stateMap[state] || { color: 'default', text: state };
    return <Tag color={config.color}>{config.text}</Tag>;
  };

  const getAlgorithmTag = (algorithm: string) => {
    const algorithmMap: { [key: string]: string } = {
      tpe: 'TPE',
      cmaes: 'CMA-ES',
      random: 'Random',
      grid: 'Grid'
    };
    return <Tag>{algorithmMap[algorithm] || algorithm}</Tag>;
  };

  const columns = [
    {
      title: <Checkbox 
        onChange={(e) => {
          if (e.target.checked) {
            setSelectedRows(experiments.map(exp => exp.id));
          } else {
            setSelectedRows([]);
          }
        }}
        aria-label="全选"
      />,
      dataIndex: 'checkbox',
      key: 'checkbox',
      width: 50,
      render: (_: any, record: Experiment) => (
        <Checkbox
          checked={selectedRows.includes(record.id)}
          onChange={(e) => {
            if (e.target.checked) {
              setSelectedRows([...selectedRows, record.id]);
            } else {
              setSelectedRows(selectedRows.filter(id => id !== record.id));
            }
          }}
        />
      )
    },
    {
      title: '实验名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
    },
    {
      title: '状态',
      dataIndex: 'state',
      key: 'state',
      render: (state: string) => getStateTag(state),
    },
    {
      title: '算法',
      dataIndex: 'algorithm',
      key: 'algorithm',
      render: (algorithm: string) => getAlgorithmTag(algorithm),
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time: string) => new Date(time).toLocaleString(),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_: any, record: Experiment) => (
        <Space>
          {record.state === 'created' && (
            <Button
              size="small"
              icon={<PlayCircleOutlined />}
              onClick={() => handleStart(record.id)}
            >
              启动
            </Button>
          )}
          {record.state === 'running' && (
            <Button
              size="small"
              icon={<PauseOutlined />}
              onClick={() => handleStop(record.id)}
            >
              停止
            </Button>
          )}
          <Button
            size="small"
            danger
            icon={<DeleteOutlined />}
            onClick={() => {
              setSelectedExperiment(record);
              setDeleteModalVisible(true);
            }}
          >
            删除
          </Button>
        </Space>
      ),
    },
  ];

  if (loading) {
    return <div data-testid="loading"><Spin size="large" /></div>;
  }

  if (!loading && experiments.length === 0) {
    return <Empty description="暂无实验" />;
  }

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <Space>
          <Search
            placeholder="搜索实验"
            onSearch={(value) => setSearchQuery(value)}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{ width: 200 }}
            prefix={<SearchOutlined />}
          />
          <Select
            placeholder="状态筛选"
            allowClear
            style={{ width: 120 }}
            onChange={(value) => setStateFilter(value)}
            aria-label="状态筛选"
          >
            <Option value="created">已创建</Option>
            <Option value="running">运行中</Option>
            <Option value="completed">已完成</Option>
            <Option value="failed">失败</Option>
            <Option value="stopped">已停止</Option>
          </Select>
          <Button icon={<ReloadOutlined />} onClick={fetchExperiments}>
            刷新
          </Button>
          <Button type="primary" icon={<PlusOutlined />}>
            创建实验
          </Button>
          {selectedRows.length > 0 && (
            <Button
              danger
              icon={<DeleteOutlined />}
              onClick={() => setDeleteModalVisible(true)}
            >
              批量删除
            </Button>
          )}
        </Space>
      </div>

      <Table
        columns={columns}
        dataSource={experiments}
        rowKey="id"
        pagination={{
          pageSize: 10,
          showSizeChanger: true,
          showTotal: (total) => `共 ${total} 条`,
        }}
      />

      <Modal
        title="确认删除"
        open={deleteModalVisible}
        onOk={() => handleDelete(selectedExperiment?.id)}
        onCancel={() => {
          setDeleteModalVisible(false);
          setSelectedExperiment(null);
        }}
        okText="确认"
        cancelText="取消"
      >
        <p>确定要删除{selectedExperiment ? `实验 "${selectedExperiment.name}"` : '选中的实验'}吗？此操作不可恢复。</p>
      </Modal>
    </div>
  );
};

export default ExperimentList;
