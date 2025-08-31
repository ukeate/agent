import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Button,
  Table,
  Tag,
  Alert,
  Space,
  Typography,
  Statistic,
  Timeline,
  Progress,
  Descriptions,
  Badge,
  Modal,
  Form,
  Input,
  Select,
  message,
  Tooltip,
  Divider
} from 'antd';
import {
  CrownOutlined,
  TeamOutlined,
  UserOutlined,
  ClockCircleOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  SyncOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { TextArea } = Input;

// Raft节点状态枚举
enum RaftState {
  FOLLOWER = 'follower',
  CANDIDATE = 'candidate',
  LEADER = 'leader'
}

interface RaftNode {
  node_id: string;
  state: RaftState;
  current_term: number;
  voted_for?: string;
  last_log_index: number;
  last_log_term: number;
  commit_index: number;
  last_applied: number;
  next_index?: Record<string, number>;
  match_index?: Record<string, number>;
  vote_count: number;
  last_heartbeat: string;
  is_active: boolean;
  network_partition: boolean;
}

interface LogEntry {
  index: number;
  term: number;
  command_type: string;
  command_data: any;
  timestamp: string;
  committed: boolean;
  applied: boolean;
}

interface ElectionEvent {
  term: number;
  candidate: string;
  voters: string[];
  result: 'won' | 'lost' | 'split';
  timestamp: string;
  duration: number;
}

const RaftConsensusPage: React.FC = () => {
  const [nodes, setNodes] = useState<RaftNode[]>([]);
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);
  const [electionHistory, setElectionHistory] = useState<ElectionEvent[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [currentTerm, setCurrentTerm] = useState(0);
  const [currentLeader, setCurrentLeader] = useState<string | null>(null);
  const [simulationSpeed, setSimulationSpeed] = useState(1000);
  const [commandModalVisible, setCommandModalVisible] = useState(false);
  const [form] = Form.useForm();

  // 初始化Raft集群
  const initializeCluster = (nodeCount: number = 5) => {
    const newNodes: RaftNode[] = Array.from({ length: nodeCount }, (_, i) => ({
      node_id: `node_${i + 1}`,
      state: RaftState.FOLLOWER,
      current_term: 0,
      last_log_index: 0,
      last_log_term: 0,
      commit_index: 0,
      last_applied: 0,
      vote_count: 0,
      last_heartbeat: new Date().toISOString(),
      is_active: true,
      network_partition: false
    }));

    setNodes(newNodes);
    setLogEntries([]);
    setElectionHistory([]);
    setCurrentTerm(0);
    setCurrentLeader(null);
  };

  // 模拟领导者选举
  const simulateElection = () => {
    const activeNodes = nodes.filter(n => n.is_active && !n.network_partition);
    if (activeNodes.length < 2) return;

    const newTerm = currentTerm + 1;
    const candidates = activeNodes.filter(n => n.state !== RaftState.LEADER);
    const candidate = candidates[Math.floor(Math.random() * candidates.length)];

    // 模拟投票过程
    const majorityNeeded = Math.floor(activeNodes.length / 2) + 1;
    const voters: string[] = [candidate.node_id]; // 候选者投给自己
    
    // 其他节点随机投票
    activeNodes.forEach(node => {
      if (node.node_id !== candidate.node_id && Math.random() > 0.3) {
        voters.push(node.node_id);
      }
    });

    const won = voters.length >= majorityNeeded;
    const result = won ? 'won' : (voters.length === 1 ? 'lost' : 'split');

    // 更新节点状态
    const updatedNodes = nodes.map(node => {
      if (!node.is_active || node.network_partition) return node;

      return {
        ...node,
        current_term: newTerm,
        state: node.node_id === candidate.node_id && won ? RaftState.LEADER : RaftState.FOLLOWER,
        voted_for: voters.includes(node.node_id) ? candidate.node_id : undefined,
        vote_count: node.node_id === candidate.node_id ? voters.length : 0,
        last_heartbeat: new Date().toISOString()
      };
    });

    setNodes(updatedNodes);
    setCurrentTerm(newTerm);
    setCurrentLeader(won ? candidate.node_id : null);

    // 记录选举历史
    const electionEvent: ElectionEvent = {
      term: newTerm,
      candidate: candidate.node_id,
      voters,
      result,
      timestamp: new Date().toISOString(),
      duration: Math.floor(Math.random() * 500) + 100
    };

    setElectionHistory(prev => [electionEvent, ...prev.slice(0, 9)]);
    message.info(`Term ${newTerm}: ${candidate.node_id} ${won ? '当选为Leader' : '选举失败'}`);
  };

  // 添加日志条目
  const appendLogEntry = (command: any) => {
    if (!currentLeader) {
      message.error('没有Leader，无法添加日志条目');
      return;
    }

    const leader = nodes.find(n => n.node_id === currentLeader);
    if (!leader) return;

    const newEntry: LogEntry = {
      index: leader.last_log_index + 1,
      term: currentTerm,
      command_type: command.type || 'custom',
      command_data: command,
      timestamp: new Date().toISOString(),
      committed: false,
      applied: false
    };

    // 更新Leader的日志索引
    const updatedNodes = nodes.map(node => 
      node.node_id === currentLeader 
        ? { ...node, last_log_index: newEntry.index, last_log_term: newEntry.term }
        : node
    );

    setNodes(updatedNodes);
    setLogEntries(prev => [newEntry, ...prev]);

    // 模拟日志复制过程
    setTimeout(() => {
      commitLogEntry(newEntry.index);
    }, 1000);

    message.success(`日志条目已添加: Index ${newEntry.index}`);
  };

  // 提交日志条目
  const commitLogEntry = (index: number) => {
    const activeNodes = nodes.filter(n => n.is_active && !n.network_partition);
    const majorityNeeded = Math.floor(activeNodes.length / 2) + 1;

    // 模拟大多数节点确认
    if (Math.random() > 0.2) { // 80%的概率成功复制
      setLogEntries(prev => 
        prev.map(entry => 
          entry.index === index 
            ? { ...entry, committed: true, applied: true }
            : entry
        )
      );

      setNodes(prev => 
        prev.map(node => 
          node.is_active 
            ? { ...node, commit_index: Math.max(node.commit_index, index), last_applied: Math.max(node.last_applied, index) }
            : node
        )
      );

      message.success(`日志条目 ${index} 已提交`);
    } else {
      message.error(`日志条目 ${index} 复制失败`);
    }
  };

  // 模拟网络分区
  const simulateNetworkPartition = (nodeId: string) => {
    setNodes(prev => 
      prev.map(node => 
        node.node_id === nodeId 
          ? { ...node, network_partition: !node.network_partition }
          : node
      )
    );

    const node = nodes.find(n => n.node_id === nodeId);
    message.info(`${nodeId} ${node?.network_partition ? '网络恢复' : '网络分区'}`);
  };

  // 模拟节点故障
  const simulateNodeFailure = (nodeId: string) => {
    setNodes(prev => 
      prev.map(node => 
        node.node_id === nodeId 
          ? { ...node, is_active: !node.is_active }
          : node
      )
    );

    const node = nodes.find(n => n.node_id === nodeId);
    message.info(`${nodeId} ${node?.is_active ? '节点恢复' : '节点故障'}`);

    // 如果Leader故障，触发新选举
    if (nodeId === currentLeader && node?.is_active) {
      setCurrentLeader(null);
      setTimeout(simulateElection, 2000);
    }
  };

  // 心跳更新
  const updateHeartbeats = () => {
    if (!currentLeader) return;

    setNodes(prev => 
      prev.map(node => 
        node.is_active && !node.network_partition
          ? { ...node, last_heartbeat: new Date().toISOString() }
          : node
      )
    );
  };

  // 自动模拟
  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      // 模拟心跳
      updateHeartbeats();

      // 随机事件
      const rand = Math.random();
      if (rand < 0.05 && !currentLeader) {
        // 5%概率触发选举（如果没有Leader）
        simulateElection();
      } else if (rand < 0.02) {
        // 2%概率随机添加日志条目
        if (currentLeader) {
          appendLogEntry({
            type: 'auto_command',
            data: { timestamp: Date.now(), value: Math.random() }
          });
        }
      }
    }, simulationSpeed);

    return () => clearInterval(interval);
  }, [isRunning, currentLeader, simulationSpeed]);

  // 初始化
  useEffect(() => {
    initializeCluster();
  }, []);

  // 节点表格列
  const nodeColumns: ColumnsType<RaftNode> = [
    {
      title: '节点ID',
      dataIndex: 'node_id',
      key: 'node_id',
      render: (id: string, record: RaftNode) => (
        <Space>
          <Badge 
            status={
              !record.is_active ? 'error' : 
              record.network_partition ? 'warning' :
              record.state === RaftState.LEADER ? 'success' : 'processing'
            } 
          />
          <Text strong={record.state === RaftState.LEADER}>{id}</Text>
          {record.state === RaftState.LEADER && <CrownOutlined style={{ color: '#faad14' }} />}
        </Space>
      )
    },
    {
      title: '状态',
      dataIndex: 'state',
      key: 'state',
      render: (state: RaftState) => {
        const stateConfig = {
          [RaftState.LEADER]: { color: 'success', icon: <CrownOutlined /> },
          [RaftState.CANDIDATE]: { color: 'warning', icon: <UserOutlined /> },
          [RaftState.FOLLOWER]: { color: 'default', icon: <TeamOutlined /> }
        };
        
        const config = stateConfig[state];
        return (
          <Tag color={config.color} icon={config.icon}>
            {state.toUpperCase()}
          </Tag>
        );
      }
    },
    {
      title: '任期',
      dataIndex: 'current_term',
      key: 'current_term'
    },
    {
      title: '日志索引',
      dataIndex: 'last_log_index',
      key: 'last_log_index'
    },
    {
      title: '提交索引',
      dataIndex: 'commit_index',
      key: 'commit_index'
    },
    {
      title: '投票数',
      dataIndex: 'vote_count',
      key: 'vote_count',
      render: (count: number, record: RaftNode) => 
        record.state === RaftState.CANDIDATE ? <Tag color="blue">{count}</Tag> : '-'
    },
    {
      title: '最后心跳',
      dataIndex: 'last_heartbeat',
      key: 'last_heartbeat',
      render: (time: string) => {
        const diff = Date.now() - new Date(time).getTime();
        const color = diff > 5000 ? 'red' : diff > 2000 ? 'orange' : 'green';
        return <Text style={{ color }}>{Math.floor(diff / 1000)}s前</Text>;
      }
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record: RaftNode) => (
        <Space>
          <Button
            size="small"
            type={record.is_active ? 'primary' : 'default'}
            danger={record.is_active}
            onClick={() => simulateNodeFailure(record.node_id)}
          >
            {record.is_active ? '故障' : '恢复'}
          </Button>
          <Button
            size="small"
            type={record.network_partition ? 'default' : 'primary'}
            onClick={() => simulateNetworkPartition(record.node_id)}
          >
            {record.network_partition ? '恢复网络' : '网络分区'}
          </Button>
        </Space>
      )
    }
  ];

  // 日志条目表格列
  const logColumns: ColumnsType<LogEntry> = [
    {
      title: '索引',
      dataIndex: 'index',
      key: 'index'
    },
    {
      title: '任期',
      dataIndex: 'term',
      key: 'term'
    },
    {
      title: '命令类型',
      dataIndex: 'command_type',
      key: 'command_type',
      render: (type: string) => <Tag color="blue">{type}</Tag>
    },
    {
      title: '命令数据',
      dataIndex: 'command_data',
      key: 'command_data',
      render: (data: any) => (
        <Text code ellipsis style={{ maxWidth: 200 }}>
          {JSON.stringify(data)}
        </Text>
      )
    },
    {
      title: '状态',
      key: 'status',
      render: (_, record: LogEntry) => (
        <Space>
          <Tag color={record.committed ? 'success' : 'default'}>
            {record.committed ? '已提交' : '待提交'}
          </Tag>
          {record.applied && <Tag color="green">已应用</Tag>}
        </Space>
      )
    },
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (time: string) => new Date(time).toLocaleTimeString()
    }
  ];

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <Title level={2}>Raft共识引擎</Title>
      <Paragraph>
        分布式一致性算法可视化演示，包含领导者选举、日志复制、故障恢复等核心机制。
      </Paragraph>

      {/* 集群状态总览 */}
      <Card title="集群状态" style={{ marginBottom: 24 }}>
        <Row gutter={16}>
          <Col span={4}>
            <Statistic
              title="当前任期"
              value={currentTerm}
              prefix={<ClockCircleOutlined />}
            />
          </Col>
          <Col span={4}>
            <Statistic
              title="当前Leader"
              value={currentLeader || '无'}
              prefix={<CrownOutlined />}
              valueStyle={{ color: currentLeader ? '#3f8600' : '#cf1322' }}
            />
          </Col>
          <Col span={4}>
            <Statistic
              title="活跃节点"
              value={nodes.filter(n => n.is_active).length}
              suffix={`/ ${nodes.length}`}
              prefix={<TeamOutlined />}
            />
          </Col>
          <Col span={4}>
            <Statistic
              title="日志条目"
              value={logEntries.length}
              prefix={<DatabaseOutlined />}
            />
          </Col>
          <Col span={4}>
            <Statistic
              title="已提交"
              value={logEntries.filter(e => e.committed).length}
              prefix={<CheckCircleOutlined />}
            />
          </Col>
          <Col span={4}>
            <Statistic
              title="网络分区"
              value={nodes.filter(n => n.network_partition).length}
              prefix={<WarningOutlined />}
              valueStyle={{ color: nodes.some(n => n.network_partition) ? '#cf1322' : '#3f8600' }}
            />
          </Col>
        </Row>

        <Divider />

        <Space>
          <Button
            type="primary"
            icon={isRunning ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
            onClick={() => setIsRunning(!isRunning)}
          >
            {isRunning ? '停止模拟' : '开始模拟'}
          </Button>
          <Button
            icon={<ThunderboltOutlined />}
            onClick={simulateElection}
            disabled={!nodes.some(n => n.is_active)}
          >
            触发选举
          </Button>
          <Button
            icon={<DatabaseOutlined />}
            onClick={() => setCommandModalVisible(true)}
            disabled={!currentLeader}
          >
            添加日志
          </Button>
          <Button
            icon={<ReloadOutlined />}
            onClick={() => initializeCluster(5)}
          >
            重置集群
          </Button>
          <Select
            value={simulationSpeed}
            onChange={setSimulationSpeed}
            style={{ width: 120 }}
          >
            <Option value={2000}>慢速</Option>
            <Option value={1000}>正常</Option>
            <Option value={500}>快速</Option>
          </Select>
        </Space>
      </Card>

      <Row gutter={24}>
        <Col span={16}>
          {/* 节点状态 */}
          <Card title="节点状态" style={{ marginBottom: 16 }}>
            <Table
              columns={nodeColumns}
              dataSource={nodes}
              rowKey="node_id"
              pagination={false}
              size="small"
            />
          </Card>

          {/* 日志条目 */}
          <Card title="日志条目">
            <Table
              columns={logColumns}
              dataSource={logEntries}
              rowKey="index"
              pagination={{ pageSize: 8, showSizeChanger: false }}
              size="small"
            />
          </Card>
        </Col>

        <Col span={8}>
          {/* 选举历史 */}
          <Card title="选举历史" style={{ marginBottom: 16 }}>
            {electionHistory.length === 0 ? (
              <Alert message="暂无选举记录" type="info" />
            ) : (
              <Timeline>
                {electionHistory.slice(0, 6).map((election, index) => (
                  <Timeline.Item
                    key={index}
                    color={election.result === 'won' ? 'green' : 'red'}
                    dot={
                      election.result === 'won' ? 
                        <CheckCircleOutlined /> : 
                        <ExclamationCircleOutlined />
                    }
                  >
                    <div>
                      <Text strong>Term {election.term}</Text>
                      <br />
                      <Text>候选者: {election.candidate}</Text>
                      <br />
                      <Text>投票数: {election.voters.length}</Text>
                      <br />
                      <Tag color={election.result === 'won' ? 'success' : 'error'}>
                        {election.result === 'won' ? '当选' : '失败'}
                      </Tag>
                      <br />
                      <Text type="secondary">
                        {new Date(election.timestamp).toLocaleTimeString()}
                      </Text>
                    </div>
                  </Timeline.Item>
                ))}
              </Timeline>
            )}
          </Card>

          {/* 系统健康度 */}
          <Card title="系统健康度" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text>节点可用性</Text>
                <Progress
                  percent={Math.round((nodes.filter(n => n.is_active).length / nodes.length) * 100)}
                  status={nodes.filter(n => n.is_active).length >= Math.floor(nodes.length / 2) + 1 ? 'success' : 'exception'}
                />
              </div>
              <div>
                <Text>网络连通性</Text>
                <Progress
                  percent={Math.round((nodes.filter(n => !n.network_partition).length / nodes.length) * 100)}
                  status={nodes.filter(n => !n.network_partition).length >= Math.floor(nodes.length / 2) + 1 ? 'success' : 'exception'}
                />
              </div>
              <div>
                <Text>日志一致性</Text>
                <Progress
                  percent={logEntries.length > 0 ? Math.round((logEntries.filter(e => e.committed).length / logEntries.length) * 100) : 100}
                />
              </div>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* 添加命令模态框 */}
      <Modal
        title="添加日志条目"
        visible={commandModalVisible}
        onCancel={() => {
          setCommandModalVisible(false);
          form.resetFields();
        }}
        onOk={async () => {
          try {
            const values = await form.validateFields();
            const command = {
              type: values.type,
              data: JSON.parse(values.data || '{}')
            };
            appendLogEntry(command);
            setCommandModalVisible(false);
            form.resetFields();
          } catch (error) {
            message.error('命令格式错误');
          }
        }}
      >
        <Form form={form} layout="vertical">
          <Form.Item
            label="命令类型"
            name="type"
            rules={[{ required: true, message: '请输入命令类型' }]}
          >
            <Select placeholder="选择命令类型">
              <Option value="set_value">设置值</Option>
              <Option value="increment">递增</Option>
              <Option value="delete">删除</Option>
              <Option value="custom">自定义</Option>
            </Select>
          </Form.Item>
          <Form.Item
            label="命令数据 (JSON)"
            name="data"
            rules={[
              { required: true, message: '请输入命令数据' },
              {
                validator: (_, value) => {
                  try {
                    JSON.parse(value || '{}');
                    return Promise.resolve();
                  } catch {
                    return Promise.reject(new Error('请输入有效的JSON格式'));
                  }
                }
              }
            ]}
          >
            <TextArea
              rows={4}
              placeholder='{"key": "value", "operation": "set"}'
            />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default RaftConsensusPage;