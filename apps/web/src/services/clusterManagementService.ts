import apiClient from './apiClient';

export interface AgentInfo {
  agent_id: string;
  node_id: string;
  name?: string;
  endpoint?: string;
  status: 'online' | 'offline' | 'error' | 'starting' | 'stopping';
  capabilities: string[];
  current_load: number;
  max_capacity: number;
  version: string;
  last_heartbeat: string;
  uptime: number;
  is_healthy?: boolean;
  resource_usage?: {
    cpu_usage: number;
    memory_usage: number;
    active_tasks: number;
    error_rate: number;
  };
  labels?: Record<string, string>;
  created_at?: number;
  updated_at?: number;
}

export interface ClusterStats {
  total_agents: number;
  online_agents: number;
  offline_agents: number;
  error_agents: number;
  total_capacity: number;
  used_capacity: number;
  avg_cpu_usage: number;
  avg_memory_usage: number;
  total_tasks_processed: number;
  error_rate: number;
  last_updated: string;
}

export interface AgentGroup {
  group_id: string;
  name: string;
  description?: string;
  agents: string[];
  max_agents?: number;
  min_agents: number;
  current_agents: number;
  labels?: Record<string, string>;
  created_at: string;
  updated_at: string;
}

export interface ScalingPolicy {
  policy_id: string;
  name: string;
  target_cpu_percent: number;
  target_memory_percent: number;
  scale_up_cpu_threshold: number;
  scale_up_memory_threshold: number;
  scale_down_cpu_threshold: number;
  scale_down_memory_threshold: number;
  min_instances: number;
  max_instances: number;
  cooldown_period_seconds: number;
  enabled: boolean;
  last_scaling_action?: string;
  created_at: string;
  updated_at: string;
}

export interface MetricsData {
  timestamp: string;
  cpu_usage: number;
  memory_usage: number;
  network_in: number;
  network_out: number;
  active_connections: number;
  request_rate: number;
  error_rate: number;
  response_time: number;
}

export interface AgentCreateRequest {
  name: string;
  host: string;
  port: number;
  capabilities?: string[];
  version?: string;
  config?: Record<string, any>;
  labels?: Record<string, string>;
  resource_spec?: {
    cpu_cores?: number;
    memory_mb?: number;
    disk_gb?: number;
  };
}

export interface GroupCreateRequest {
  name: string;
  description?: string;
  max_agents?: number;
  min_agents?: number;
  labels?: Record<string, string>;
}

export interface ScalingPolicyRequest {
  name: string;
  target_cpu_percent?: number;
  target_memory_percent?: number;
  scale_up_cpu_threshold?: number;
  scale_up_memory_threshold?: number;
  scale_down_cpu_threshold?: number;
  scale_down_memory_threshold?: number;
  min_instances?: number;
  max_instances?: number;
  cooldown_period_seconds?: number;
  enabled?: boolean;
}

export interface ManualScalingRequest {
  target_instances: number;
  reason?: string;
}

class ClusterManagementService {
  private baseUrl = '/cluster';

  private unwrap<T>(payload: any): T {
    if (payload && typeof payload === 'object' && 'data' in payload) {
      return payload.data as T;
    }
    return payload as T;
  }

  // 智能体管理
  async getAgents(): Promise<AgentInfo[]> {
    const response = await apiClient.get(`${this.baseUrl}/agents`);
    const data = this.unwrap<any>(response.data);
    if (Array.isArray(data)) {
      return data;
    }
    return data?.agents || [];
  }

  async getAgent(agentId: string): Promise<AgentInfo> {
    const response = await apiClient.get(`${this.baseUrl}/agents/${agentId}`);
    return this.unwrap(response.data);
  }

  async createAgent(request: AgentCreateRequest): Promise<AgentInfo> {
    const response = await apiClient.post(`${this.baseUrl}/agents`, request);
    return this.unwrap(response.data);
  }

  async updateAgent(agentId: string, updates: Partial<AgentCreateRequest>): Promise<AgentInfo> {
    const response = await apiClient.put(`${this.baseUrl}/agents/${agentId}`, updates);
    return this.unwrap(response.data);
  }

  async deleteAgent(agentId: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/agents/${agentId}`);
  }

  async startAgent(agentId: string): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(`${this.baseUrl}/agents/${agentId}/start`);
    return this.unwrap(response.data);
  }

  async stopAgent(agentId: string): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(`${this.baseUrl}/agents/${agentId}/stop`);
    return this.unwrap(response.data);
  }

  async restartAgent(agentId: string): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(`${this.baseUrl}/agents/${agentId}/restart`);
    return this.unwrap(response.data);
  }

  // 集群统计
  async getClusterStats(): Promise<ClusterStats> {
    const response = await apiClient.get(`${this.baseUrl}/status`);
    const stats = this.unwrap<any>(response.data) || {};
    const resource = stats.resource_usage || {};
    const totalAgents = stats.total_agents || 0;
    const runningAgents = stats.running_agents ?? stats.healthy_agents ?? 0;
    const healthyAgents = stats.healthy_agents ?? runningAgents;
    const failedAgents = Math.max(0, totalAgents - healthyAgents);
    const totalRequests = resource.total_requests || 0;
    const failedRequests = resource.failed_requests || 0;
    const errorRate = totalRequests > 0 ? (failedRequests / totalRequests) * 100 : 0;
    const updatedAtSeconds = typeof stats.updated_at === 'number' ? stats.updated_at : Date.now() / 1000;

    return {
      total_agents: totalAgents,
      online_agents: runningAgents,
      offline_agents: Math.max(0, totalAgents - runningAgents),
      error_agents: failedAgents,
      total_capacity: 100,
      used_capacity: resource.cpu_usage_percent ?? 0,
      avg_cpu_usage: resource.cpu_usage_percent ?? 0,
      avg_memory_usage: resource.memory_usage_percent ?? 0,
      total_tasks_processed: resource.active_tasks ?? 0,
      error_rate: Math.round(errorRate * 100) / 100,
      last_updated: new Date(updatedAtSeconds * 1000).toISOString(),
    };
  }

  async getClusterHealth(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    issues: string[];
    recommendations: string[];
  }> {
    const response = await apiClient.get(`${this.baseUrl}/health`);
    const health = this.unwrap<any>(response.data) || {};
    const issues: string[] = [];
    const failedAgents = health.failed_agents ?? 0;
    const healthScore = typeof health.health_score === 'number' ? health.health_score : 1;
    if (failedAgents > 0) {
      issues.push(`存在${failedAgents}个异常智能体`);
    }
    if (healthScore < 0.9) {
      issues.push('集群健康评分偏低');
    }
    const status: 'healthy' | 'degraded' | 'unhealthy' =
      healthScore >= 0.9 ? 'healthy' : healthScore >= 0.7 ? 'degraded' : 'unhealthy';
    const recommendations = issues.length
      ? ['检查异常智能体', '查看资源使用情况与健康检查日志']
      : [];
    return { status, issues, recommendations };
  }

  // 分组管理
  async getGroups(): Promise<AgentGroup[]> {
    const response = await apiClient.get(`${this.baseUrl}/groups`);
    const data = this.unwrap<any>(response.data);
    if (Array.isArray(data)) {
      return data.map((group: any) => ({
        group_id: group.group_id,
        name: group.name,
        description: group.description,
        agents: group.agent_ids || group.agents || [],
        max_agents: group.max_agents,
        min_agents: group.min_agents ?? 0,
        current_agents: group.agent_count ?? group.current_agents ?? (group.agent_ids ? group.agent_ids.length : 0),
        labels: group.labels,
        created_at: new Date((group.created_at ?? Date.now() / 1000) * 1000).toISOString(),
        updated_at: new Date((group.updated_at ?? Date.now() / 1000) * 1000).toISOString(),
      }));
    }
    return (data?.groups || []).map((group: any) => ({
      group_id: group.group_id,
      name: group.name,
      description: group.description,
      agents: group.agent_ids || group.agents || [],
      max_agents: group.max_agents,
      min_agents: group.min_agents ?? 0,
      current_agents: group.agent_count ?? group.current_agents ?? (group.agent_ids ? group.agent_ids.length : 0),
      labels: group.labels,
      created_at: new Date((group.created_at ?? Date.now() / 1000) * 1000).toISOString(),
      updated_at: new Date((group.updated_at ?? Date.now() / 1000) * 1000).toISOString(),
    }));
  }

  async getGroup(groupId: string): Promise<AgentGroup> {
    const response = await apiClient.get(`${this.baseUrl}/groups/${groupId}`);
    return this.unwrap(response.data);
  }

  async createGroup(request: GroupCreateRequest): Promise<AgentGroup> {
    const response = await apiClient.post(`${this.baseUrl}/groups`, request);
    return this.unwrap(response.data);
  }

  async updateGroup(groupId: string, updates: Partial<GroupCreateRequest>): Promise<AgentGroup> {
    const response = await apiClient.put(`${this.baseUrl}/groups/${groupId}`, updates);
    return this.unwrap(response.data);
  }

  async deleteGroup(groupId: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/groups/${groupId}`);
  }

  async addAgentToGroup(groupId: string, agentId: string): Promise<void> {
    await apiClient.post(`${this.baseUrl}/groups/${groupId}/agents/${agentId}`);
  }

  async removeAgentFromGroup(groupId: string, agentId: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/groups/${groupId}/agents/${agentId}`);
  }

  // 扩缩容管理
  async getScalingPolicies(): Promise<ScalingPolicy[]> {
    const response = await apiClient.get(`${this.baseUrl}/scaling/policies`);
    const data = this.unwrap<any>(response.data);
    if (Array.isArray(data)) {
      return data;
    }
    return data?.policies || [];
  }

  async getScalingPolicy(policyId: string): Promise<ScalingPolicy> {
    const response = await apiClient.get(`${this.baseUrl}/scaling/policies/${policyId}`);
    return this.unwrap(response.data);
  }

  async createScalingPolicy(request: ScalingPolicyRequest): Promise<ScalingPolicy> {
    const response = await apiClient.post(`${this.baseUrl}/scaling/policies`, request);
    return this.unwrap(response.data);
  }

  async updateScalingPolicy(policyId: string, updates: Partial<ScalingPolicyRequest>): Promise<ScalingPolicy> {
    const response = await apiClient.put(`${this.baseUrl}/scaling/policies/${policyId}`, updates);
    return this.unwrap(response.data);
  }

  async deleteScalingPolicy(policyId: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/scaling/policies/${policyId}`);
  }

  async manualScale(groupId: string, request: ManualScalingRequest): Promise<{
    success: boolean;
    current_instances: number;
    target_instances: number;
    message: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/scaling/groups/${groupId}/manual`, request);
    const data = this.unwrap<any>(response.data) || {};
    const success = typeof data.success === 'boolean' ? data.success : Boolean(response.data?.success);
    return {
      success,
      current_instances: data.current_instances ?? 0,
      target_instances: data.target_instances ?? request.target_instances,
      message: data.message ?? (success ? 'ok' : 'failed'),
    };
  }

  async getScalingHistory(groupId?: string, days?: number): Promise<Array<{
    timestamp: string;
    action: 'scale_up' | 'scale_down';
    from_instances: number;
    to_instances: number;
    reason: string;
    success: boolean;
  }>> {
    const params = { group_id: groupId, limit: days ? days * 24 : undefined };
    const response = await apiClient.get(`${this.baseUrl}/scaling/history`, { params });
    const data = this.unwrap<any>(response.data);
    if (!Array.isArray(data)) {
      return data?.history || [];
    }
    return data.map((event: any) => ({
      timestamp: event?.end_time ? new Date(event.end_time * 1000).toISOString() : new Date().toISOString(),
      action: event?.decision?.action || 'scale_up',
      from_instances: event?.decision?.current_instances || 0,
      to_instances: event?.decision?.target_instances || 0,
      reason: event?.decision?.reason || '',
      success: Boolean(event?.success),
    }));
  }

  // 监控指标
  async getMetrics(agentId?: string, duration?: number): Promise<MetricsData[]> {
    const metricNames = agentId
      ? [
          'total_requests',
          'avg_response_time',
          'error_rate',
          'cpu_usage_percent',
          'memory_usage_percent',
          'network_io_mbps',
          'active_tasks',
        ]
      : [
          'cluster_total_requests',
          'cluster_avg_response_time',
          'cluster_error_rate',
          'cluster_cpu_usage',
          'cluster_memory_usage',
          'cluster_network_io',
          'cluster_active_tasks',
        ];

    const response = await apiClient.post(`${this.baseUrl}/metrics/query`, {
      agent_id: agentId,
      duration_seconds: duration || 3600,
      metric_names: metricNames,
    });
    const data = this.unwrap<any>(response.data) || {};
    const keys = agentId
      ? {
          total: 'total_requests',
          response: 'avg_response_time',
          error: 'error_rate',
          cpu: 'cpu_usage_percent',
          memory: 'memory_usage_percent',
          network: 'network_io_mbps',
          active: 'active_tasks',
        }
      : {
          total: 'cluster_total_requests',
          response: 'cluster_avg_response_time',
          error: 'cluster_error_rate',
          cpu: 'cluster_cpu_usage',
          memory: 'cluster_memory_usage',
          network: 'cluster_network_io',
          active: 'cluster_active_tasks',
        };

    const seriesList = [
      data[keys.total],
      data[keys.response],
      data[keys.error],
      data[keys.cpu],
      data[keys.memory],
      data[keys.network],
      data[keys.active],
    ].filter(Array.isArray);

    const baseSeries = seriesList.reduce((best: any[], current: any[]) => {
      if (!best || current.length > best.length) return current;
      return best;
    }, [] as any[]);

    const toSeconds = (value: any) => {
      if (typeof value === 'number') return value;
      if (typeof value === 'string') {
        const parsed = Date.parse(value);
        if (!Number.isNaN(parsed)) return parsed / 1000;
      }
      return Date.now() / 1000;
    };

    const getValue = (series: any[], index: number) => {
      if (!Array.isArray(series) || series.length === 0) return { value: 0, timestamp: Date.now() / 1000 };
      const item = series[index] ?? series[series.length - 1];
      return {
        value: typeof item?.value === 'number' ? item.value : 0,
        timestamp: toSeconds(item?.timestamp),
      };
    };

    return baseSeries.map((_: any, index: number) => {
      const total = getValue(data[keys.total], index);
      const prevTotal = index > 0 ? getValue(data[keys.total], index - 1) : total;
      const deltaTime = Math.max(0, total.timestamp - prevTotal.timestamp);
      const requestRate = deltaTime > 0 ? Math.max(0, total.value - prevTotal.value) / deltaTime : 0;

      const responseTime = getValue(data[keys.response], index).value;
      const errorRate = getValue(data[keys.error], index).value;
      const cpuUsage = getValue(data[keys.cpu], index).value;
      const memoryUsage = getValue(data[keys.memory], index).value;
      const networkIo = getValue(data[keys.network], index).value;
      const activeTasks = getValue(data[keys.active], index).value;

      return {
        timestamp: new Date(total.timestamp * 1000).toISOString(),
        cpu_usage: cpuUsage,
        memory_usage: memoryUsage,
        network_in: networkIo,
        network_out: networkIo,
        active_connections: activeTasks,
        request_rate: requestRate,
        error_rate: errorRate,
        response_time: responseTime,
      };
    });
  }

  async getAgentMetrics(agentId: string, metricNames?: string[], duration?: number): Promise<{
    agent_id: string;
    metrics: Record<string, Array<{ timestamp: string; value: number }>>;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/metrics/query`, {
      agent_id: agentId,
      metric_names: metricNames,
      duration_seconds: duration || 3600,
    });
    const data = this.unwrap<any>(response.data) || {};
    return { agent_id: agentId, metrics: data };
  }

  async getAggregatedMetrics(groupId?: string, aggregation?: 'avg' | 'sum' | 'max' | 'min'): Promise<{
    cpu_usage: number;
    memory_usage: number;
    task_count: number;
    error_rate: number;
    response_time: number;
  }> {
    const params = {
      agent_id: groupId,
      duration_seconds: 3600,
    };
    const response = await apiClient.get(`${this.baseUrl}/metrics/summary`, { params });
    const summary = this.unwrap<any>(response.data) || {};
    const pick = (name: string) => {
      const entry = summary[name];
      if (!entry) return 0;
      const value = entry[aggregation || 'avg'] ?? entry.average ?? entry.latest ?? 0;
      return typeof value === 'number' ? value : 0;
    };
    const cpuKey = groupId ? 'cpu_usage_percent' : 'cluster_cpu_usage';
    const memoryKey = groupId ? 'memory_usage_percent' : 'cluster_memory_usage';
    const taskKey = groupId ? 'active_tasks' : 'cluster_active_tasks';
    const errorKey = groupId ? 'error_rate' : 'cluster_error_rate';
    const responseKey = groupId ? 'avg_response_time' : 'cluster_avg_response_time';
    return {
      cpu_usage: pick(cpuKey),
      memory_usage: pick(memoryKey),
      task_count: pick(taskKey),
      error_rate: pick(errorKey),
      response_time: pick(responseKey),
    };
  }

  // 实时状态
  async getRealtimeStatus(): Promise<{
    agents: Array<{
      agent_id: string;
      status: string;
      cpu: number;
      memory: number;
      tasks: number;
    }>;
    timestamp: string;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/realtime/status`);
    return response.data;
  }

  // 任务分配
  async getTaskDistribution(): Promise<{
    distribution: Record<string, number>;
    total_tasks: number;
    avg_per_agent: number;
    std_deviation: number;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/tasks/distribution`);
    return response.data;
  }

  async rebalanceTasks(groupId?: string): Promise<{
    success: boolean;
    tasks_moved: number;
    from_agents: string[];
    to_agents: string[];
  }> {
    const params = groupId ? { group_id: groupId } : {};
    const response = await apiClient.post(`${this.baseUrl}/tasks/rebalance`, {}, { params });
    return response.data;
  }
}

export const clusterManagementService = new ClusterManagementService();
