import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState, useEffect, useRef } from 'react';
import {
import { logger } from '../utils/logger'
  Card,
  Tabs,
  Button,
  Input,
  Select,
  Row,
  Col,
  Typography,
  Space,
  Tag,
  Alert,
  Progress,
  Divider,
  Badge,
  message,
  Table,
  Modal,
  Form,
  Avatar,
  Spin,
  Radio
} from 'antd';
import {
  NodeIndexOutlined,
  ShareAltOutlined,
  ExperimentOutlined,
  SyncOutlined,
  ArrowDownOutlined,
  UsergroupAddOutlined,
  UserOutlined,
  LinkOutlined,
  ClusterOutlined,
  RadarChartOutlined,
  BarChartOutlined
} from '@ant-design/icons';
import * as d3 from 'd3';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;
const { TabPane } = Tabs;
const { Option } = Select;

// 网络节点和边的数据类型
interface NetworkNode {
  user_id: string;
  emotional_influence: number;
  connection_strength: Record<string, number>;
  emotion_state: {
    emotions: Record<string, number>;
    intensity: number;
    confidence: number;
  };
  role: string;
  x?: number;
  y?: number;
}

interface NetworkEdge {
  source: string;
  target: string;
  weight: number;
  emotion_sync: number;
  interaction_count: number;
}

interface EmotionNetwork {
  network_id: string;
  nodes: Record<string, NetworkNode>;
  edges: Record<string, number>;
  clusters: string[][];
  central_nodes: string[];
  influence_paths: Record<string, string[][]>;
  network_cohesion: number;
  polarization_level: number;
}

interface NetworkAnalytics {
  total_connections: number;
  average_strength: number;
  clustering_coefficient: number;
  network_diameter: number;
  community_count: number;
  influence_distribution: Record<string, number>;
  emotion_synchrony: number;
  stability_score: number;
}

// API 客户端
const socialNetworkApi = {
  async buildEmotionNetwork(payload: any) {
    const response = await apiFetch(buildApiUrl(`/api/v1/social-emotion/analyze`), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const result = await response.json();
    return {
      success: true,
      data: {
        emotion_network: result.results?.network_analysis || null
      }
    };
  }
};

// 已移除本地随机生成逻辑，依赖真实API

const SocialNetworkEmotionMapPage: React.FC = () => {
  const [currentNetwork, setCurrentNetwork] = useState<EmotionNetwork | null>(null);
  const [networkAnalytics, setNetworkAnalytics] = useState<NetworkAnalytics | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedLayout, setSelectedLayout] = useState('force');
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [showBuildModal, setShowBuildModal] = useState(false);
  const [viewMode, setViewMode] = useState<'network' | 'clusters' | 'influence'>('network');
  const [lastBuildPayload, setLastBuildPayload] = useState<any | null>(null);
  
  const [buildForm] = Form.useForm();
  const networkRef = useRef<SVGSVGElement>(null);
  const simulation = useRef<d3.Simulation<any, any> | null>(null);

  const layoutOptions = [
    { value: 'force', label: '力导引布局' },
    { value: 'circular', label: '圆形布局' },
    { value: 'hierarchical', label: '层次布局' },
    { value: 'grid', label: '网格布局' }
  ];

  const parseJsonInput = (value: string, fieldLabel: string) => {
    if (!value) {
      throw new Error(`${fieldLabel}不能为空`);
    }
    try {
      return JSON.parse(value);
    } catch (error) {
      throw new Error(`${fieldLabel}不是有效的JSON`);
    }
  };

  const computeNetworkAnalytics = (network: EmotionNetwork): NetworkAnalytics => {
    const edgeEntries = Object.entries(network.edges);
    const totalConnections = edgeEntries.length;
    const averageStrength = totalConnections
      ? edgeEntries.reduce((sum, [, weight]) => sum + weight, 0) / totalConnections
      : 0;

    const nodeIds = Object.keys(network.nodes);
    const adjacency: Record<string, Set<string>> = {};
    nodeIds.forEach((id) => {
      adjacency[id] = new Set();
    });

    edgeEntries.forEach(([key]) => {
      const [source, target] = key.split('-');
      if (source && target) {
        adjacency[source]?.add(target);
        adjacency[target]?.add(source);
      }
    });

    const clusteringCoefficients = nodeIds.map((id) => {
      const neighbors = Array.from(adjacency[id] || []);
      if (neighbors.length < 2) return 0;
      let connections = 0;
      for (let i = 0; i < neighbors.length; i += 1) {
        for (let j = i + 1; j < neighbors.length; j += 1) {
          if (adjacency[neighbors[i]]?.has(neighbors[j])) {
            connections += 1;
          }
        }
      }
      const possible = (neighbors.length * (neighbors.length - 1)) / 2;
      return possible ? connections / possible : 0;
    });

    const clusteringCoefficient = clusteringCoefficients.length
      ? clusteringCoefficients.reduce((sum, value) => sum + value, 0) / clusteringCoefficients.length
      : 0;

    const bfsDistances = (start: string) => {
      const distances: Record<string, number> = {};
      const queue: string[] = [];
      distances[start] = 0;
      queue.push(start);
      while (queue.length) {
        const current = queue.shift() as string;
        const currentDistance = distances[current];
        adjacency[current]?.forEach((neighbor) => {
          if (distances[neighbor] === undefined) {
            distances[neighbor] = currentDistance + 1;
            queue.push(neighbor);
          }
        });
      }
      return distances;
    };

    let networkDiameter = 0;
    nodeIds.forEach((id) => {
      const distances = bfsDistances(id);
      const maxDistance = Object.values(distances).reduce((max, value) => Math.max(max, value), 0);
      networkDiameter = Math.max(networkDiameter, maxDistance);
    });

    const influenceCounts = { low: 0, medium: 0, high: 0 };
    nodeIds.forEach((id) => {
      const influence = network.nodes[id]?.emotional_influence || 0;
      if (influence >= 0.66) {
        influenceCounts.high += 1;
      } else if (influence >= 0.33) {
        influenceCounts.medium += 1;
      } else {
        influenceCounts.low += 1;
      }
    });
    const influenceDistribution = nodeIds.length
      ? {
          low: influenceCounts.low / nodeIds.length,
          medium: influenceCounts.medium / nodeIds.length,
          high: influenceCounts.high / nodeIds.length
        }
      : { low: 0, medium: 0, high: 0 };

    const dominantScores = nodeIds.map((id) => {
      const emotions = network.nodes[id]?.emotion_state?.emotions || {};
      const values = Object.values(emotions);
      return values.length ? Math.max(...values) : 0;
    });
    const avgDominant = dominantScores.length
      ? dominantScores.reduce((sum, value) => sum + value, 0) / dominantScores.length
      : 0;
    const variance = dominantScores.length
      ? dominantScores.reduce((sum, value) => sum + (value - avgDominant) ** 2, 0) / dominantScores.length
      : 0;
    const emotionSynchrony = Math.max(0, Math.min(1, 1 - variance));

    return {
      total_connections: totalConnections,
      average_strength: averageStrength,
      clustering_coefficient: clusteringCoefficient,
      network_diameter: networkDiameter,
      community_count: network.clusters?.length || 0,
      influence_distribution: influenceDistribution,
      emotion_synchrony: emotionSynchrony,
      stability_score: Math.max(0, Math.min(1, 1 - (network.polarization_level || 0)))
    };
  };

  useEffect(() => {
    if (lastBuildPayload && !currentNetwork) {
      loadNetworkData();
    }
  }, [lastBuildPayload]);

  useEffect(() => {
    if (currentNetwork && networkRef.current) {
      renderNetwork();
    }
  }, [currentNetwork, selectedLayout, viewMode]);

  useEffect(() => {
    if (currentNetwork) {
      setNetworkAnalytics(computeNetworkAnalytics(currentNetwork));
    } else {
      setNetworkAnalytics(null);
    }
  }, [currentNetwork]);

  const loadNetworkData = async () => {
    if (!lastBuildPayload) {
      message.warning('请先构建网络');
      return;
    }
    setLoading(true);
    try {
      const networkResult = await socialNetworkApi.buildEmotionNetwork(lastBuildPayload);
      if (networkResult.data?.emotion_network) {
        setCurrentNetwork(networkResult.data.emotion_network);
        message.success('网络数据加载完成');
      } else {
        setCurrentNetwork(null);
        message.warning('未返回网络数据');
      }
    } catch (error) {
      logger.error('加载失败:', error);
      message.error('数据加载失败');
    } finally {
      setLoading(false);
    }
  };

  const renderNetwork = () => {
    if (!networkRef.current || !currentNetwork) return;

    const svg = d3.select(networkRef.current);
    svg.selectAll("*").remove();

    const width = 800;
    const height = 600;
    const margin = { top: 20, right: 20, bottom: 20, left: 20 };

    svg.attr("width", width).attr("height", height);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    // 准备节点和边数据
    const nodes = Object.values(currentNetwork.nodes).map(node => ({
      ...node,
      id: node.user_id
    }));

    const links = Object.entries(currentNetwork.edges).map(([edgeKey, weight]) => {
      const [source, target] = edgeKey.split('-');
      return { source, target, weight };
    });

    // 颜色比例尺
    const colorScale = d3.scaleOrdinal()
      .domain(['influencer', 'supporter', 'connector', 'neutral', 'independent'])
      .range(['#ff4d4f', '#52c41a', '#1890ff', '#d9d9d9', '#722ed1']);

    // 根据布局类型设置simulation
    if (simulation.current) {
      simulation.current.stop();
    }

    if (selectedLayout === 'force') {
      simulation.current = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id((d: any) => d.id).strength(0.1))
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter((width - margin.left - margin.right) / 2, (height - margin.top - margin.bottom) / 2))
        .force("collision", d3.forceCollide().radius(30));
    } else if (selectedLayout === 'circular') {
      const radius = Math.min(width, height) / 3;
      const angleStep = (2 * Math.PI) / nodes.length;
      nodes.forEach((node, i) => {
        node.x = (width - margin.left - margin.right) / 2 + radius * Math.cos(i * angleStep);
        node.y = (height - margin.top - margin.bottom) / 2 + radius * Math.sin(i * angleStep);
      });
    }

    // 绘制连线
    const link = g.append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(links)
      .enter().append("line")
      .attr("class", "link")
      .attr("stroke", "#999")
      .attr("stroke-opacity", 0.6)
      .attr("stroke-width", (d: any) => Math.sqrt(d.weight * 10));

    // 绘制节点
    const node = g.append("g")
      .attr("class", "nodes")
      .selectAll("g")
      .data(nodes)
      .enter().append("g")
      .attr("class", "node")
      .call(drag(simulation.current as any));

    // 添加圆形节点
    node.append("circle")
      .attr("r", (d: any) => 15 + d.emotional_influence * 20)
      .attr("fill", (d: any) => colorScale(d.role) as string)
      .attr("stroke", (d: any) => selectedNode === d.id ? "#000" : "#fff")
      .attr("stroke-width", (d: any) => selectedNode === d.id ? 3 : 2)
      .on("click", (event, d: any) => {
        setSelectedNode(d.id === selectedNode ? null : d.id);
      })
      .on("mouseover", function(event, d: any) {
        d3.select(this).attr("r", (d.emotional_influence * 20) + 20);
        
        // 显示tooltip
        const tooltip = svg.append("g")
          .attr("class", "tooltip")
          .attr("transform", `translate(${d.x + margin.left + 20},${d.y + margin.top})`);
        
        tooltip.append("rect")
          .attr("width", 120)
          .attr("height", 60)
          .attr("fill", "white")
          .attr("stroke", "#ccc")
          .attr("rx", 4);
          
        tooltip.append("text")
          .attr("x", 8)
          .attr("y", 16)
          .text(d.user_id)
          .style("font-weight", "bold");
          
        tooltip.append("text")
          .attr("x", 8)
          .attr("y", 32)
          .text(`角色: ${d.role}`)
          .style("font-size", "12px");
          
        tooltip.append("text")
          .attr("x", 8)
          .attr("y", 48)
          .text(`影响力: ${(d.emotional_influence * 100).toFixed(0)}%`)
          .style("font-size", "12px");
      })
      .on("mouseout", function(event, d: any) {
        d3.select(this).attr("r", 15 + d.emotional_influence * 20);
        svg.select(".tooltip").remove();
      });

    // 添加标签
    node.append("text")
      .text((d: any) => d.user_id)
      .attr("dx", 0)
      .attr("dy", 4)
      .attr("text-anchor", "middle")
      .style("font-size", "12px")
      .style("fill", "white")
      .style("font-weight", "bold");

    // 根据视图模式添加额外的视觉元素
    if (viewMode === 'clusters') {
      drawClusters(g, currentNetwork.clusters, nodes);
    } else if (viewMode === 'influence') {
      drawInfluencePaths(g, currentNetwork.influence_paths, nodes);
    }

    // 如果使用力导引布局，更新位置
    if (selectedLayout === 'force' && simulation.current) {
      simulation.current.on("tick", () => {
        link
          .attr("x1", (d: any) => d.source.x)
          .attr("y1", (d: any) => d.source.y)
          .attr("x2", (d: any) => d.target.x)
          .attr("y2", (d: any) => d.target.y);

        node.attr("transform", (d: any) => `translate(${d.x},${d.y})`);
      });
    } else {
      // 静态布局
      link
        .attr("x1", (d: any) => nodes.find(n => n.id === d.source)?.x)
        .attr("y1", (d: any) => nodes.find(n => n.id === d.source)?.y)
        .attr("x2", (d: any) => nodes.find(n => n.id === d.target)?.x)
        .attr("y2", (d: any) => nodes.find(n => n.id === d.target)?.y);

      node.attr("transform", (d: any) => `translate(${d.x},${d.y})`);
    }
  };

  const drawClusters = (g: any, clusters: string[][], nodes: any[]) => {
    clusters.forEach((cluster, i) => {
      const clusterNodes = nodes.filter(n => cluster.includes(n.id));
      if (clusterNodes.length < 2) return;

      const hull = d3.polygonHull(clusterNodes.map(d => [d.x, d.y]));
      if (!hull) return;

      g.append("path")
        .datum(hull)
        .attr("class", `cluster-${i}`)
        .attr("d", (d: any) => `M${d.join("L")}Z`)
        .attr("fill", d3.schemeSet3[i % d3.schemeSet3.length])
        .attr("fill-opacity", 0.2)
        .attr("stroke", d3.schemeSet3[i % d3.schemeSet3.length])
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "5,5");
    });
  };

  const drawInfluencePaths = (g: any, influencePaths: Record<string, string[][]>, nodes: any[]) => {
    Object.entries(influencePaths).forEach(([influencer, paths]) => {
      paths.forEach((path, i) => {
        const pathNodes = path.map(nodeId => nodes.find(n => n.id === nodeId)).filter(Boolean);
        if (pathNodes.length < 2) return;

        const line = d3.line()
          .x((d: any) => d.x)
          .y((d: any) => d.y)
          .curve(d3.curveBasis);

        g.append("path")
          .datum(pathNodes)
          .attr("class", `influence-path-${i}`)
          .attr("d", line)
          .attr("fill", "none")
          .attr("stroke", "#ff7875")
          .attr("stroke-width", 3)
          .attr("stroke-opacity", 0.7)
          .attr("marker-end", "url(#arrowhead)");
      });
    });

    // 添加箭头标记
    const defs = g.append("defs");
    defs.append("marker")
      .attr("id", "arrowhead")
      .attr("viewBox", "-0 -5 10 10")
      .attr("refX", 8)
      .attr("refY", 0)
      .attr("orient", "auto")
      .attr("markerWidth", 8)
      .attr("markerHeight", 8)
      .append("path")
      .attr("d", "M 0,-5 L 10,0 L 0,5")
      .attr("fill", "#ff7875");
  };

  const drag = (simulation: d3.Simulation<any, any>) => {
    function dragstarted(event: any) {
      if (!event.active && simulation) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }
    
    function dragged(event: any) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }
    
    function dragended(event: any) {
      if (!event.active && simulation) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }
    
    return d3.drag()
      .on("start", dragstarted)
      .on("drag", dragged)
      .on("end", dragended);
  };

  const buildNewNetwork = async (values: any) => {
    setLoading(true);
    try {
      const payload = {
        user_id: values.user_id?.trim(),
        session_id: values.session_id?.trim() || undefined,
        emotion_data: parseJsonInput(values.emotion_data, '情感数据'),
        social_context: {
          scenario: values.scenario || 'casual',
          cultural_context: values.cultural_context?.trim() || undefined,
          session_data: parseJsonInput(values.session_data, '会话数据'),
          interaction_history: parseJsonInput(values.interaction_history, '交互历史')
        },
        analysis_type: ['network_analysis'],
        cultural_context: values.cultural_context?.trim() || undefined,
        privacy_consent: true
      };
      const result = await socialNetworkApi.buildEmotionNetwork(payload);
      
      if (result.data?.emotion_network) {
        setCurrentNetwork(result.data.emotion_network);
        setLastBuildPayload(payload);
        message.success('网络构建完成');
        setShowBuildModal(false);
      }
    } catch (error) {
      const errorMessage = (error as Error)?.message || '网络构建失败';
      message.error(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const performCommunityDetection = async () => {
    if (!currentNetwork) return;
    message.success(`检测到 ${currentNetwork.clusters.length} 个社区`);
  };

  const renderOverviewCards = () => (
    <Row gutter={16}>
      <Col span={6}>
        <Card>
          <div style={{ textAlign: 'center' }}>
            <UsergroupAddOutlined style={{ fontSize: 24, color: '#1890ff', marginBottom: 8 }} />
            <div style={{ fontSize: 24, fontWeight: 'bold', color: '#1890ff' }}>
              {currentNetwork ? Object.keys(currentNetwork.nodes).length : 0}
            </div>
            <div style={{ color: '#8c8c8c' }}>节点数量</div>
          </div>
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <div style={{ textAlign: 'center' }}>
            <LinkOutlined style={{ fontSize: 24, color: '#52c41a', marginBottom: 8 }} />
            <div style={{ fontSize: 24, fontWeight: 'bold', color: '#52c41a' }}>
              {networkAnalytics?.total_connections || 0}
            </div>
            <div style={{ color: '#8c8c8c' }}>连接数量</div>
          </div>
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <div style={{ textAlign: 'center' }}>
            <ClusterOutlined style={{ fontSize: 24, color: '#fa8c16', marginBottom: 8 }} />
            <div style={{ fontSize: 24, fontWeight: 'bold', color: '#fa8c16' }}>
              {networkAnalytics?.community_count || 0}
            </div>
            <div style={{ color: '#8c8c8c' }}>社区数量</div>
          </div>
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <div style={{ textAlign: 'center' }}>
            <RadarChartOutlined style={{ fontSize: 24, color: '#722ed1', marginBottom: 8 }} />
            <div style={{ fontSize: 20, fontWeight: 'bold', color: '#722ed1' }}>
              {currentNetwork ? Math.round(currentNetwork.network_cohesion * 100) : 0}%
            </div>
            <div style={{ color: '#8c8c8c' }}>网络凝聚力</div>
          </div>
        </Card>
      </Col>
    </Row>
  );

  const renderNetworkView = () => (
    <Card 
      title={
        <span>
          <NodeIndexOutlined style={{ marginRight: 8 }} />
          社交网络情感地图
        </span>
      }
      extra={
        <Space>
          <span>视图模式:</span>
          <Radio.Group value={viewMode} onChange={e => setViewMode(e.target.value)} size="small">
            <Radio.Button value="network">网络</Radio.Button>
            <Radio.Button value="clusters">聚类</Radio.Button>
            <Radio.Button value="influence">影响</Radio.Button>
          </Radio.Group>
          <span>布局:</span>
          <Select
            value={selectedLayout}
            onChange={setSelectedLayout}
            style={{ width: 120 }}
            size="small"
          >
            {layoutOptions.map(option => (
              <Option key={option.value} value={option.value}>
                {option.label}
              </Option>
            ))}
          </Select>
        </Space>
      }
    >
      {currentNetwork ? (
        <div style={{ position: 'relative' }}>
          <svg ref={networkRef}></svg>
          {selectedNode && (
            <div style={{
              position: 'absolute',
              top: 10,
              right: 10,
              width: 200,
              backgroundColor: 'white',
              border: '1px solid #d9d9d9',
              borderRadius: 4,
              padding: 12
            }}>
              <Title level={5}>节点详情</Title>
              <div>
                <Text strong>用户:</Text> {selectedNode}
              </div>
              <div>
                <Text strong>角色:</Text> {currentNetwork.nodes[selectedNode]?.role}
              </div>
              <div>
                <Text strong>影响力:</Text>
                <Progress
                  percent={Math.round((currentNetwork.nodes[selectedNode]?.emotional_influence || 0) * 100)}
                  size="small"
                  style={{ marginTop: 4 }}
                />
              </div>
            </div>
          )}
        </div>
      ) : (
        <div style={{ textAlign: 'center', padding: 60 }}>
          <Spin spinning={loading}>
            <Text type="secondary">暂无网络数据</Text>
          </Spin>
        </div>
      )}
    </Card>
  );

  const renderNodeList = () => {
    if (!currentNetwork) return null;

    const columns = [
      {
        title: '用户',
        dataIndex: 'user_id',
        key: 'user_id',
        render: (userId: string) => (
          <Space>
            <Avatar size="small" style={{ backgroundColor: '#87d068' }}>
              {userId.charAt(0).toUpperCase()}
            </Avatar>
            <span>{userId}</span>
          </Space>
        )
      },
      {
        title: '角色',
        dataIndex: 'role',
        key: 'role',
        render: (role: string) => {
          const roleColors = {
            influencer: 'red',
            supporter: 'green',
            connector: 'blue',
            neutral: 'gray',
            independent: 'purple'
          };
          return (
            <Tag color={roleColors[role as keyof typeof roleColors]}>
              {role}
            </Tag>
          );
        }
      },
      {
        title: '情感影响力',
        dataIndex: 'emotional_influence',
        key: 'emotional_influence',
        render: (influence: number) => (
          <Progress
            percent={Math.round(influence * 100)}
            size="small"
            strokeColor="#1890ff"
            style={{ width: 120 }}
          />
        )
      },
      {
        title: '情感状态',
        dataIndex: 'emotion_state',
        key: 'emotion_state',
        render: (state: any) => {
          const dominantEmotion = Object.entries(state.emotions)
            .sort(([,a], [,b]) => (b as number) - (a as number))[0];
          return (
            <Space>
              <Tag color="purple">{dominantEmotion[0]}</Tag>
              <Text style={{ fontSize: '12px' }}>
                {((dominantEmotion[1] as number) * 100).toFixed(0)}%
              </Text>
            </Space>
          );
        }
      },
      {
        title: '连接数',
        dataIndex: 'connection_strength',
        key: 'connections',
        render: (connections: Record<string, number>) => (
          <Badge count={Object.keys(connections).length} style={{ backgroundColor: '#52c41a' }} />
        )
      }
    ];

    const nodeData = Object.values(currentNetwork.nodes);

    return (
      <Card title={
        <span>
          <UserOutlined style={{ marginRight: 8 }} />
          节点详细信息 ({nodeData.length})
        </span>
      }>
        <Table
          columns={columns}
          dataSource={nodeData}
          rowKey="user_id"
          pagination={{ pageSize: 10 }}
          size="small"
        />
      </Card>
    );
  };

  const renderInfluencePaths = () => {
    if (!currentNetwork || !currentNetwork.influence_paths) return null;

    return (
      <Card title={
        <span>
          <ShareAltOutlined style={{ marginRight: 8 }} />
          影响传播路径
        </span>
      }>
        {Object.entries(currentNetwork.influence_paths).map(([influencer, paths]) => (
          <div key={influencer} style={{ marginBottom: 16 }}>
            <Title level={5}>
              <Avatar size="small" style={{ backgroundColor: '#f56a00' }}>
                {influencer.charAt(0)}
              </Avatar>
              <span style={{ marginLeft: 8 }}>{influencer} 的影响路径</span>
            </Title>
            {paths.map((path, index) => (
              <div key={index} style={{ marginBottom: 8 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  {path.map((nodeId, nodeIndex) => (
                    <React.Fragment key={nodeId}>
                      <Tag color="blue">{nodeId}</Tag>
                      {nodeIndex < path.length - 1 && (
                        <ArrowDownOutlined style={{ color: '#999' }} />
                      )}
                    </React.Fragment>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ))}
      </Card>
    );
  };

  const renderClusters = () => {
    if (!currentNetwork) return null;

    return (
      <Card title={
        <span>
          <ClusterOutlined style={{ marginRight: 8 }} />
          社区聚类 ({currentNetwork.clusters.length})
        </span>
      }>
        {currentNetwork.clusters.map((cluster, index) => (
          <div key={index} style={{ marginBottom: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <Title level={5}>社区 {index + 1}</Title>
              <Badge count={cluster.length} style={{ backgroundColor: '#1890ff' }} />
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {cluster.map(member => (
                <Tag key={member} color="blue">{member}</Tag>
              ))}
            </div>
          </div>
        ))}
      </Card>
    );
  };

  const renderNetworkMetrics = () => (
    <Card title={
      <span>
        <BarChartOutlined style={{ marginRight: 8 }} />
        网络指标
      </span>
    }>
      {networkAnalytics ? (
        <div>
          <Row gutter={[16, 16]}>
            <Col span={12}>
              <div>
                <Text strong>平均连接强度:</Text>
                <Progress
                  percent={Math.round(networkAnalytics.average_strength * 100)}
                  size="small"
                  style={{ marginTop: 4 }}
                />
              </div>
            </Col>
            <Col span={12}>
              <div>
                <Text strong>聚类系数:</Text>
                <Progress
                  percent={Math.round(networkAnalytics.clustering_coefficient * 100)}
                  size="small"
                  style={{ marginTop: 4 }}
                  strokeColor="#52c41a"
                />
              </div>
            </Col>
            <Col span={12}>
              <div>
                <Text strong>情感同步性:</Text>
                <Progress
                  percent={Math.round(networkAnalytics.emotion_synchrony * 100)}
                  size="small"
                  style={{ marginTop: 4 }}
                  strokeColor="#722ed1"
                />
              </div>
            </Col>
            <Col span={12}>
              <div>
                <Text strong>稳定性分数:</Text>
                <Progress
                  percent={Math.round(networkAnalytics.stability_score * 100)}
                  size="small"
                  style={{ marginTop: 4 }}
                  strokeColor="#fa8c16"
                />
              </div>
            </Col>
          </Row>
          
          <Divider />
          
          <div>
            <Text strong>网络直径:</Text>
            <span style={{ marginLeft: 8, fontSize: '16px', fontWeight: 'bold' }}>
              {networkAnalytics.network_diameter}
            </span>
          </div>
          
          <div style={{ marginTop: 16 }}>
            <Text strong>影响力分布:</Text>
            <div style={{ marginTop: 8 }}>
              {Object.entries(networkAnalytics.influence_distribution).map(([level, value]) => (
                <div key={level} style={{ marginBottom: 8 }}>
                  <Text style={{ textTransform: 'capitalize', marginRight: 8 }}>{level}:</Text>
                  <Progress
                    percent={Math.round((value as number) * 100)}
                    size="small"
                    style={{ width: 200 }}
                  />
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <Text type="secondary">暂无指标数据</Text>
      )}
    </Card>
  );

  const renderBuildModal = () => (
    <Modal
      title="构建社交网络"
      open={showBuildModal}
      onCancel={() => setShowBuildModal(false)}
      footer={[
        <Button key="cancel" onClick={() => setShowBuildModal(false)}>
          取消
        </Button>,
        <Button 
          key="build" 
          type="primary" 
          loading={loading}
          onClick={() => buildForm.submit()}
        >
          构建网络
        </Button>
      ]}
      width={600}
    >
      <Form
        form={buildForm}
        layout="vertical"
        onFinish={buildNewNetwork}
      >
        <Alert
          message="社交网络构建"
          description="基于情感数据构建社交网络，分析用户间的情感连接和影响关系"
          type="info"
          showIcon
          style={{ marginBottom: 24 }}
        />

        <Form.Item
          label="用户ID"
          name="user_id"
          rules={[{ required: true, message: '请输入用户ID' }]}
        >
          <Input placeholder="输入用于构建网络的用户ID" />
        </Form.Item>

        <Form.Item
          label="会话ID"
          name="session_id"
        >
          <Input placeholder="可选，用于标识会话" />
        </Form.Item>

        <Form.Item
          label="场景类型"
          name="scenario"
          rules={[{ required: true, message: '请选择场景类型' }]}
          initialValue="casual"
        >
          <Select placeholder="选择社交场景">
            <Option value="meeting">正式会议</Option>
            <Option value="casual">日常沟通</Option>
            <Option value="brainstorm">头脑风暴</Option>
            <Option value="conflict">冲突协调</Option>
            <Option value="presentation">公开表达</Option>
          </Select>
        </Form.Item>

        <Form.Item
          label="文化背景"
          name="cultural_context"
        >
          <Input placeholder="可选，例如地区或文化标签" />
        </Form.Item>

        <Form.Item
          label="情感数据（JSON）"
          name="emotion_data"
          rules={[{ required: true, message: '请输入情感数据JSON' }]}
        >
          <TextArea
            rows={4}
            placeholder="请输入情感数据JSON，包含emotions、intensity、confidence等字段"
          />
        </Form.Item>

        <Form.Item
          label="会话数据（JSON数组）"
          name="session_data"
          rules={[{ required: true, message: '请输入会话数据JSON数组' }]}
        >
          <TextArea
            rows={4}
            placeholder="请输入会话数据JSON数组，包含user_id、timestamp、emotion_data等字段"
          />
        </Form.Item>

        <Form.Item
          label="交互历史（JSON数组）"
          name="interaction_history"
          rules={[{ required: true, message: '请输入交互历史JSON数组' }]}
        >
          <TextArea
            rows={4}
            placeholder="请输入交互历史JSON数组，包含参与者交互的结构化数据"
          />
        </Form.Item>
      </Form>
    </Modal>
  );

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={2}>
          <NodeIndexOutlined style={{ marginRight: 12, color: '#1890ff' }} />
          社交网络情感地图
        </Title>
        <Space>
          <Button 
            type="primary" 
            icon={<ExperimentOutlined />}
            onClick={() => setShowBuildModal(true)}
          >
            构建网络
          </Button>
          <Button 
            icon={<ClusterOutlined />}
            onClick={performCommunityDetection}
          >
            社区检测
          </Button>
          <Button 
            icon={<SyncOutlined />} 
            loading={loading}
            onClick={loadNetworkData}
          >
            刷新
          </Button>
        </Space>
      </div>

      <div style={{ marginBottom: 24 }}>
        {renderOverviewCards()}
      </div>

      <Tabs defaultActiveKey="network-view">
        <TabPane tab="网络视图" key="network-view">
          {renderNetworkView()}
        </TabPane>

        <TabPane tab="节点分析" key="node-analysis">
          {renderNodeList()}
        </TabPane>

        <TabPane tab="影响路径" key="influence-paths">
          {renderInfluencePaths()}
        </TabPane>

        <TabPane tab="社区聚类" key="clusters">
          {renderClusters()}
        </TabPane>

        <TabPane tab="网络指标" key="metrics">
          {renderNetworkMetrics()}
        </TabPane>
      </Tabs>

      {renderBuildModal()}
    </div>
  );
};

export default SocialNetworkEmotionMapPage;
