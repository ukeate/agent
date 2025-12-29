import apiClient from './apiClient';

export interface Entity {
  id: string;
  canonical_form: string;
  entity_type: string;
  text?: string;
  confidence: number;
  language?: string;
  linked_entity?: string;
  metadata: Record<string, any>;
  created_at: string;
  updated_at?: string;
}

export interface Relation {
  id: string;
  source_entity_id: string;
  target_entity_id: string;
  relation_type: string;
  confidence: number;
  context: string;
  source_sentence: string;
  evidence: string[];
  metadata: Record<string, any>;
  created_at: string;
}

export interface CreateEntityRequest {
  canonical_form: string;
  entity_type: string;
  text?: string;
  confidence?: number;
  language?: string;
  linked_entity?: string;
  metadata?: Record<string, any>;
}

export interface UpdateEntityRequest {
  canonical_form?: string;
  confidence?: number;
  language?: string;
  linked_entity?: string;
  metadata?: Record<string, any>;
}

export interface CreateRelationRequest {
  source_entity_id: string;
  target_entity_id: string;
  relation_type: string;
  confidence?: number;
  context: string;
  source_sentence: string;
  evidence?: string[];
  metadata?: Record<string, any>;
}

export interface EntitySearchRequest {
  canonical_form_contains?: string;
  entity_type?: string;
  confidence_gte?: number;
  confidence_lte?: number;
  created_after?: string;
  created_before?: string;
  limit?: number;
  skip?: number;
}

export interface CustomQueryRequest {
  query: string;
  parameters?: Record<string, any>;
  read_only?: boolean;
}

export interface QueryResult {
  columns: string[];
  data: any[][];
  stats: {
    nodes_created: number;
    nodes_deleted: number;
    relationships_created: number;
    relationships_deleted: number;
    properties_set: number;
    execution_time: number;
    records_returned: number;
  };
}

export interface GraphSchema {
  entity_types: Array<{
    name: string;
    properties: Record<string, string>;
    count: number;
  }>;
  relation_types: Array<{
    name: string;
    properties: Record<string, string>;
    count: number;
  }>;
}

export interface GraphStats {
  total_entities: number;
  total_relations: number;
  entity_types: Record<string, number>;
  relation_types: Record<string, number>;
  average_relations_per_entity: number;
  graph_density: number;
  last_updated: string;
}

export interface QueryTemplate {
  id: string;
  name: string;
  description: string;
  query: string;
  category: string;
  parameters?: string[];
  created_at: string;
  usage_count: number;
}

export interface Migration {
  id: string;
  name: string;
  description: string;
  version: string;
  migration_type: string;
  up_statements: string[];
  down_statements: string[];
  dependencies: string[];
  created_at: string;
  status?: string;
}

export interface MigrationRecord {
  migration_id: string;
  status: string;
  started_at: string;
  completed_at?: string | null;
  error_message?: string | null;
  rollback_at?: string | null;
  execution_time_ms?: number | null;
  affected_nodes: number;
  affected_relationships: number;
}

export interface CreateMigrationRequest {
  name: string;
  description: string;
  version?: string;
  migration_type: string;
  up_statements: string[];
  down_statements?: string[];
  dependencies?: string[];
}

class KnowledgeGraphService {
  private baseUrl = '/knowledge-graph';

  // Entity Operations
  async createEntity(request: CreateEntityRequest): Promise<Entity> {
    const response = await apiClient.post(`${this.baseUrl}/entities`, request);
    return response.data;
  }

  async getEntity(entityId: string): Promise<Entity> {
    const response = await apiClient.get(`${this.baseUrl}/entities/${entityId}`);
    return response.data;
  }

  async updateEntity(entityId: string, request: UpdateEntityRequest): Promise<Entity> {
    const response = await apiClient.put(`${this.baseUrl}/entities/${entityId}`, request);
    return response.data;
  }

  async deleteEntity(entityId: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/entities/${entityId}`);
  }

  async searchEntities(request: EntitySearchRequest): Promise<Entity[]> {
    const response = await apiClient.post(`${this.baseUrl}/entities/search`, request);
    return response.data.entities;
  }

  // Relation Operations
  async createRelation(request: CreateRelationRequest): Promise<Relation> {
    const response = await apiClient.post(`${this.baseUrl}/relations`, request);
    return response.data;
  }

  async getRelation(relationId: string): Promise<Relation> {
    const response = await apiClient.get(`${this.baseUrl}/relations/${relationId}`);
    return response.data;
  }

  async deleteRelation(relationId: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/relations/${relationId}`);
  }

  async getEntityRelations(entityId: string, direction?: 'incoming' | 'outgoing' | 'both'): Promise<Relation[]> {
    const params = direction ? { direction } : {};
    const response = await apiClient.get(`${this.baseUrl}/entities/${entityId}/relations`, { params });
    return response.data.relations;
  }

  // Query Operations
  async executeQuery(request: CustomQueryRequest): Promise<QueryResult> {
    const response = await apiClient.post(`${this.baseUrl}/query`, request);
    return response.data;
  }

  async getQueryTemplates(category?: string): Promise<QueryTemplate[]> {
    const params = category ? { category } : {};
    const response = await apiClient.get(`${this.baseUrl}/query/templates`, { params });
    return response.data.templates || [];
  }

  async saveQueryTemplate(template: Omit<QueryTemplate, 'id' | 'created_at' | 'usage_count'>): Promise<QueryTemplate> {
    const response = await apiClient.post(`${this.baseUrl}/query/templates`, template);
    return response.data;
  }

  async deleteQueryTemplate(templateId: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/query/templates/${templateId}`);
  }

  // Schema Operations
  async getSchema(): Promise<GraphSchema> {
    const response = await apiClient.get(`${this.baseUrl}/schema`);
    return response.data;
  }

  async updateSchema(schema: Partial<GraphSchema>): Promise<GraphSchema> {
    const response = await apiClient.put(`${this.baseUrl}/schema`, schema);
    return response.data;
  }

  // Statistics
  async getStats(): Promise<GraphStats> {
    const response = await apiClient.get(`${this.baseUrl}/stats`);
    return response.data;
  }

  // Batch Operations
  async batchUpsertEntities(entities: CreateEntityRequest[], conflictStrategy?: string): Promise<{
    success: number;
    failed: number;
    errors: Array<{ entity: CreateEntityRequest; error: string }>;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/entities/batch`, {
      entities,
      conflict_strategy: conflictStrategy || 'merge_highest_confidence'
    });
    return response.data;
  }

  // Path Finding
  async findShortestPath(sourceId: string, targetId: string, maxDepth?: number): Promise<{
    path: Array<Entity | Relation>;
    length: number;
  }> {
    const params = maxDepth ? { max_depth: maxDepth } : {};
    const response = await apiClient.get(`${this.baseUrl}/paths/shortest`, {
      params: { source_id: sourceId, target_id: targetId, ...params }
    });
    return response.data;
  }

  async findAllPaths(sourceId: string, targetId: string, maxDepth?: number, limit?: number): Promise<Array<{
    path: Array<Entity | Relation>;
    length: number;
  }>> {
    const params = {
      source_id: sourceId,
      target_id: targetId,
      max_depth: maxDepth || 5,
      limit: limit || 10
    };
    const response = await apiClient.get(`${this.baseUrl}/paths/all`, { params });
    return response.data.paths;
  }

  // Subgraph Extraction
  async getSubgraph(entityId: string, depth?: number): Promise<{
    entities: Entity[];
    relations: Relation[];
  }> {
    const params = depth ? { depth } : {};
    const response = await apiClient.get(`${this.baseUrl}/subgraph/${entityId}`, { params });
    return response.data;
  }

  // Quality Control APIs
  async getQualityIssues(): Promise<Array<{
    issue_id: string;
    issue_type: string;
    description: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    affected_entities: string[];
    recommended_actions: string[];
    confidence: number;
    detected_at: string;
  }>> {
    const response = await apiClient.get(`${this.baseUrl}/quality/issues`);
    return response.data.issues;
  }

  async getQualityReport(): Promise<{
    total_entities: number;
    total_relations: number;
    quality_score: number;
    issues_summary: Record<string, number>;
    recommendations: string[];
  }> {
    const response = await apiClient.get(`${this.baseUrl}/quality/report`);
    return response.data;
  }

  // Performance Monitoring APIs
  async getSlowQueries(): Promise<Array<{
    query: string;
    execution_time: number;
    timestamp: string;
    frequency: number;
  }>> {
    const response = await apiClient.get(`${this.baseUrl}/performance/slow-queries`);
    return response.data.queries;
  }

  async clearPerformanceCache(): Promise<{ message: string }> {
    const response = await apiClient.delete(`${this.baseUrl}/performance/cache`);
    return response.data;
  }

  // Statistics API
  async getStatistics(): Promise<GraphStats> {
    const response = await apiClient.get(`${this.baseUrl}/statistics`);
    return response.data;
  }

  // Admin Schema Initialization
  async initializeSchema(): Promise<{ message: string; entities_created: number; relations_created: number }> {
    const response = await apiClient.post(`${this.baseUrl}/admin/schema/initialize`);
    return response.data;
  }

  // Upsert Operations
  async upsertEntity(request: CreateEntityRequest): Promise<Entity> {
    const response = await apiClient.post(`${this.baseUrl}/upsert-entity`, request);
    return response.data;
  }

  async batchUpsert(entities: CreateEntityRequest[]): Promise<{
    success: number;
    failed: number;
    errors: Array<{ entity: CreateEntityRequest; error: string }>;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/batch-upsert`, { entities });
    return response.data;
  }
  async extractSubgraph(entityIds: string[], depth: number = 1): Promise<{
    entities: Entity[];
    relations: Relation[];
  }> {
    const response = await apiClient.post(`${this.baseUrl}/subgraph`, {
      entity_ids: entityIds,
      depth
    });
    return response.data;
  }

  // Export/Import
  async exportGraph(format: 'json' | 'csv' | 'graphml' = 'json'): Promise<Blob> {
    const response = await apiClient.get(`${this.baseUrl}/export`, {
      params: { format },
      responseType: 'blob'
    });
    return response.data;
  }

  async importGraph(file: File, format: 'json' | 'csv' | 'graphml'): Promise<{
    entities_imported: number;
    relations_imported: number;
    errors: string[];
  }> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('format', format);
    
    const response = await apiClient.post(`${this.baseUrl}/import`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  }

  // Quality Management
  async validateGraph(): Promise<{
    valid: boolean;
    issues: Array<{
      type: string;
      severity: 'error' | 'warning' | 'info';
      message: string;
      entity_ids?: string[];
    }>;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/validate`);
    return response.data;
  }

  async cleanupDuplicates(): Promise<{
    duplicates_found: number;
    duplicates_merged: number;
    entities_affected: string[];
  }> {
    const response = await apiClient.post(`${this.baseUrl}/cleanup/duplicates`);
    return response.data;
  }

  // ========== Knowledge Graph Reasoning APIs ==========
  async queryReasoning(request: {
    query: string;
    reasoning_strategy?: string;
    max_depth?: number;
    confidence_threshold?: number;
  }): Promise<{
    result: any;
    reasoning_trace: Array<{
      step: number;
      operation: string;
      confidence: number;
      explanation: string;
    }>;
    total_confidence: number;
  }> {
    const response = await apiClient.post('/kg-reasoning/query', request);
    return response.data;
  }

  async batchReasoning(requests: Array<{
    query_id: string;
    query: string;
    reasoning_strategy?: string;
  }>): Promise<{
    batch_id: string;
    results: Array<{
      query_id: string;
      result: any;
      status: 'success' | 'failed';
      error?: string;
    }>;
  }> {
    const response = await apiClient.post('/kg-reasoning/batch', { queries: requests });
    return response.data;
  }

  async getStrategyPerformance(): Promise<Array<{
    strategy_name: string;
    avg_execution_time: number;
    accuracy_score: number;
    usage_count: number;
    success_rate: number;
  }>> {
    const response = await apiClient.get('/kg-reasoning/strategies/performance');
    return response.data.strategies;
  }

  async explainReasoning(request: {
    query: string;
    result_id?: string;
    explain_level: 'basic' | 'detailed' | 'technical';
  }): Promise<{
    explanation: string;
    reasoning_steps: Array<{
      step_type: string;
      input: any;
      output: any;
      confidence: number;
      time_ms: number;
    }>;
    visualizations?: Array<{
      type: string;
      data: any;
    }>;
  }> {
    const response = await apiClient.post('/kg-reasoning/explain', request);
    return response.data;
  }

  // ========== GraphRAG APIs ==========
  async graphragQuery(request: {
    question: string;
    context_depth?: number;
    max_tokens?: number;
    include_sources?: boolean;
  }): Promise<{
    query_id: string;
    answer: string;
    sources: Array<{
      entity_id: string;
      entity_name: string;
      relevance_score: number;
      content_snippet: string;
    }>;
    reasoning_path: Array<string>;
    confidence: number;
  }> {
    const response = await apiClient.post('/graphrag/query', request);
    return response.data;
  }

  async graphragReasoningQuery(request: {
    question: string;
    reasoning_type: 'causal' | 'comparative' | 'temporal' | 'compositional';
    evidence_threshold?: number;
  }): Promise<{
    query_id: string;
    answer: string;
    reasoning_type: string;
    evidence_chain: Array<{
      premise: string;
      conclusion: string;
      confidence: number;
      supporting_entities: string[];
    }>;
    alternative_answers?: Array<{
      answer: string;
      confidence: number;
      reasoning: string;
    }>;
  }> {
    const response = await apiClient.post('/graphrag/query/reasoning', request);
    return response.data;
  }

  async getGraphragQueryResult(queryId: string): Promise<{
    query_id: string;
    status: 'processing' | 'completed' | 'failed';
    result?: any;
    error?: string;
    started_at: string;
    completed_at?: string;
  }> {
    const response = await apiClient.get(`/graphrag/query/${queryId}`);
    return response.data;
  }

  async multiSourceFusion(request: {
    sources: Array<{
      source_id: string;
      source_type: string;
      weight?: number;
    }>;
    fusion_strategy: 'majority_vote' | 'weighted_average' | 'confidence_based';
    conflict_resolution: 'prefer_higher_confidence' | 'aggregate' | 'flag_conflicts';
  }): Promise<{
    fusion_id: string;
    fused_knowledge: Array<{
      entity_id: string;
      fused_attributes: Record<string, any>;
      source_contributions: Record<string, number>;
      confidence: number;
    }>;
    conflicts_detected: Array<{
      entity_id: string;
      attribute: string;
      conflicting_values: Array<{
        value: any;
        source_id: string;
        confidence: number;
      }>;
    }>;
  }> {
    const response = await apiClient.post('/graphrag/fusion/multi-source', request);
    return response.data;
  }

  async conflictResolution(request: {
    conflicts: Array<{
      entity_id: string;
      attribute: string;
      conflicting_values: Array<{
        value: any;
        source_id: string;
        confidence: number;
      }>;
    }>;
    resolution_strategy: string;
    manual_overrides?: Record<string, any>;
  }): Promise<{
    resolved_conflicts: Array<{
      entity_id: string;
      attribute: string;
      resolved_value: any;
      resolution_reason: string;
      confidence: number;
    }>;
    unresolved_conflicts: Array<{
      entity_id: string;
      attribute: string;
      reason: string;
    }>;
  }> {
    const response = await apiClient.post('/graphrag/fusion/conflict-resolution', request);
    return response.data;
  }

  async getConsistencyReport(): Promise<{
    overall_score: number;
    consistency_metrics: {
      temporal_consistency: number;
      logical_consistency: number;
      referential_integrity: number;
    };
    inconsistencies: Array<{
      type: string;
      description: string;
      affected_entities: string[];
      severity: 'low' | 'medium' | 'high';
    }>;
    recommendations: string[];
  }> {
    const response = await apiClient.get('/graphrag/fusion/consistency');
    return response.data;
  }

  async getPerformanceComparison(params?: {
    time_range?: string;
    comparison_metrics?: string[];
  }): Promise<{
    comparison_data: Array<{
      metric_name: string;
      current_value: number;
      previous_value: number;
      change_percentage: number;
      trend: 'improving' | 'declining' | 'stable';
    }>;
    performance_insights: string[];
    bottlenecks: Array<{
      component: string;
      issue: string;
      impact: string;
    }>;
  }> {
    const response = await apiClient.get('/graphrag/performance/comparison', { params });
    return response.data;
  }

  async runPerformanceBenchmark(request: {
    benchmark_type: 'query_performance' | 'fusion_accuracy' | 'reasoning_speed';
    test_scenarios: Array<{
      scenario_name: string;
      parameters: Record<string, any>;
    }>;
    iterations?: number;
  }): Promise<{
    benchmark_id: string;
    results: Array<{
      scenario_name: string;
      avg_execution_time: number;
      accuracy_score?: number;
      throughput?: number;
      resource_usage: {
        cpu_percent: number;
        memory_mb: number;
      };
    }>;
    overall_score: number;
    recommendations: string[];
  }> {
    const response = await apiClient.post('/graphrag/performance/benchmark', request);
    return response.data;
  }

  async debugExplain(request: {
    query_id?: string;
    query?: string;
    explain_components: string[];
    detail_level: 'summary' | 'detailed' | 'verbose';
  }): Promise<{
    explanation: {
      query_processing: string;
      graph_traversal: string;
      fusion_process: string;
      reasoning_steps: string;
    };
    performance_breakdown: Record<string, number>;
    intermediate_results: Array<{
      step: string;
      result: any;
      timing: number;
    }>;
    suggestions: string[];
  }> {
    const response = await apiClient.post('/graphrag/debug/explain', request);
    return response.data;
  }

  async getDebugTrace(queryId: string): Promise<{
    query_id: string;
    trace_data: Array<{
      timestamp: string;
      component: string;
      operation: string;
      input_data: any;
      output_data: any;
      execution_time: number;
      status: 'success' | 'error' | 'warning';
    }>;
    performance_summary: {
      total_time: number;
      bottlenecks: Array<{
        component: string;
        time_spent: number;
        percentage: number;
      }>;
    };
    errors: Array<{
      timestamp: string;
      component: string;
      error_message: string;
      stack_trace?: string;
    }>;
  }> {
    const response = await apiClient.get(`/graphrag/debug/trace/${queryId}`);
    return response.data;
  }

  // ========== LangGraph Features APIs ==========
  async demoContextApi(request: {
    demo_type: 'context-api' | 'durability' | 'caching' | 'hooks';
    parameters?: Record<string, any>;
  }): Promise<{
    demo_id: string;
    demo_type: string;
    result: any;
    execution_time: number;
    features_demonstrated: string[];
    code_example?: string;
  }> {
    const demoEndpoints = {
      'context-api': '/langgraph/context-api/demo',
      'durability': '/langgraph/durability/demo',
      'caching': '/langgraph/caching/demo',
      'hooks': '/langgraph/hooks/demo'
    };
    
    const endpoint = demoEndpoints[request.demo_type];
    const response = await apiClient.post(endpoint, request.parameters || {});
    return response.data;
  }

  async clearLangGraphCache(): Promise<{
    cache_cleared: boolean;
    items_removed: number;
    cache_size_before_mb: number;
    cache_size_after_mb: number;
  }> {
    const response = await apiClient.post('/langgraph/cache/clear');
    return response.data;
  }

  async executeLangGraphDemo(request: {
    workflow_type: string;
    input_data: any;
    config?: Record<string, any>;
  }): Promise<{
    execution_id: string;
    status: 'running' | 'completed' | 'failed';
    result?: any;
    workflow_state: any;
    execution_time?: number;
  }> {
    const response = await apiClient.post('/langgraph/execute-demo', request);
    return response.data;
  }

  async completeLangGraphDemo(request: {
    execution_id: string;
    additional_input?: any;
  }): Promise<{
    execution_id: string;
    final_result: any;
    execution_summary: {
      total_steps: number;
      total_time: number;
      success: boolean;
    };
    state_history: Array<{
      step: number;
      state: any;
      timestamp: string;
    }>;
  }> {
    const response = await apiClient.post('/langgraph/complete-demo', request);
    return response.data;
  }

  // ========== RAG GraphRAG Integration ==========
  async ragGraphragQuery(request: {
    question: string;
    document_collection?: string;
    hybrid_search?: boolean;
    rerank_results?: boolean;
  }): Promise<{
    answer: string;
    sources: Array<{
      document_id: string;
      chunk_id: string;
      relevance_score: number;
      content: string;
    }>;
    graph_context: Array<{
      entity_id: string;
      entity_type: string;
      relationship_context: string;
    }>;
    response_quality: {
      relevance_score: number;
      completeness_score: number;
      factuality_score: number;
    };
  }> {
    const response = await apiClient.post('/rag/graphrag/query', request);
    return response.data;
  }

  // ========== Migration APIs ==========
  async listMigrations(): Promise<Migration[]> {
    const response = await apiClient.get(`${this.baseUrl}/migrations`);
    return response.data.migrations || [];
  }

  async getMigrationRecords(): Promise<MigrationRecord[]> {
    const response = await apiClient.get(`${this.baseUrl}/migrations/records`);
    return response.data.records || [];
  }

  async applyMigration(migrationId: string): Promise<MigrationRecord> {
    const response = await apiClient.post(`${this.baseUrl}/migrations/${migrationId}/apply`);
    return response.data;
  }

  async applyAllMigrations(): Promise<MigrationRecord[]> {
    const response = await apiClient.post(`${this.baseUrl}/migrations/apply-all`);
    return response.data.records || [];
  }

  async rollbackMigration(migrationId: string): Promise<MigrationRecord> {
    const response = await apiClient.post(`${this.baseUrl}/migrations/${migrationId}/rollback`);
    return response.data;
  }
}

export const knowledgeGraphService = new KnowledgeGraphService();
