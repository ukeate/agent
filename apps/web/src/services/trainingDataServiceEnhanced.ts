import apiClient from './apiClient';

// 增强的训练数据管理服务，补充原有服务的高级功能
class TrainingDataServiceEnhanced {
  private baseUrl = '/training-data';

  // ========== 高级数据质量分析功能 ==========
  async performAdvancedQualityAnalysis(datasetId: string, options: {
    analysis_types: string[];
    thresholds: Record<string, number>;
    sample_size?: number;
    include_recommendations: boolean;
  }): Promise<{
    analysis_id: string;
    quality_metrics: {
      completeness: number;
      accuracy: number;
      consistency: number;
      validity: number;
      uniqueness: number;
      semantic_quality: number;
    };
    detailed_issues: Array<{
      issue_type: string;
      severity: 'low' | 'medium' | 'high' | 'critical';
      count: number;
      description: string;
      examples: string[];
      suggested_fixes: string[];
    }>;
    recommendations: {
      quality_improvement_steps: string[];
      processing_optimizations: string[];
      annotation_guidelines: string[];
    };
    comparison_baseline?: any;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/datasets/${datasetId}/advanced-quality-analysis`, options);
    return response.data;
  }

  async getBatchQualityMetrics(datasetIds: string[]): Promise<{
    datasets: Array<{
      dataset_id: string;
      quality_summary: any;
      comparison_rank: number;
    }>;
    benchmark_metrics: any;
    recommendations: string[];
  }> {
    const response = await apiClient.post(`${this.baseUrl}/batch-quality-metrics`, { dataset_ids: datasetIds });
    return response.data;
  }

  async trackQualityTrends(datasetId: string, timeRange: {
    start_date: string;
    end_date: string;
    granularity: 'hour' | 'day' | 'week' | 'month';
  }): Promise<{
    trend_data: Array<{
      timestamp: string;
      quality_score: number;
      issues_count: number;
      processed_records: number;
    }>;
    trend_analysis: {
      overall_trend: 'improving' | 'declining' | 'stable';
      significant_changes: any[];
      predictions: any;
    };
  }> {
    const response = await apiClient.post(`${this.baseUrl}/datasets/${datasetId}/quality-trends`, timeRange);
    return response.data;
  }

  // ========== 智能数据标注功能 ==========
  async createIntelligentAnnotationTask(taskConfig: {
    name: string;
    dataset_id: string;
    annotation_type: 'classification' | 'ner' | 'relation_extraction' | 'summarization';
    ai_assistance_level: 'none' | 'suggestions' | 'pre_annotation' | 'active_learning';
    quality_requirements: {
      minimum_agreement: number;
      confidence_threshold: number;
      review_percentage: number;
    };
    workflow_settings: {
      annotation_rounds: number;
      validation_strategy: string;
      conflict_resolution: string;
    };
    guidelines: string;
    examples: any[];
  }): Promise<{
    task_id: string;
    estimated_completion_time: number;
    cost_estimate: number;
    annotator_assignments: any[];
  }> {
    const response = await apiClient.post(`${this.baseUrl}/intelligent-annotation-tasks`, taskConfig);
    return response.data;
  }

  async getAnnotationInsights(taskId: string): Promise<{
    productivity_metrics: {
      annotations_per_hour: number;
      quality_consistency: number;
      learning_curve_progress: number;
    };
    quality_insights: {
      agreement_patterns: any[];
      common_errors: any[];
      improvement_suggestions: string[];
    };
    annotator_performance: Array<{
      annotator_id: string;
      performance_score: number;
      strengths: string[];
      areas_for_improvement: string[];
    }>;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/annotation-tasks/${taskId}/insights`);
    return response.data;
  }

  async optimizeAnnotationWorkflow(taskId: string, optimizationGoals: {
    target_metric: 'speed' | 'quality' | 'cost_efficiency';
    constraints: Record<string, any>;
  }): Promise<{
    optimization_plan: {
      workflow_changes: any[];
      resource_reallocation: any[];
      estimated_improvement: any;
    };
    implementation_steps: string[];
  }> {
    const response = await apiClient.post(`${this.baseUrl}/annotation-tasks/${taskId}/optimize`, optimizationGoals);
    return response.data;
  }

  // ========== 高级数据预处理管道 ==========
  async createAdvancedPreprocessingPipeline(pipelineConfig: {
    name: string;
    description: string;
    input_sources: string[];
    processing_steps: Array<{
      step_id: string;
      processor_type: string;
      parameters: Record<string, any>;
      parallel_execution: boolean;
      error_handling: 'skip' | 'retry' | 'fail_pipeline';
    }>;
    output_config: {
      format: string;
      validation_rules: any[];
      quality_gates: any[];
    };
    scheduling: {
      trigger_type: 'manual' | 'scheduled' | 'event_driven';
      schedule_expression?: string;
      dependencies?: string[];
    };
  }): Promise<{
    pipeline_id: string;
    execution_plan: any;
    resource_requirements: any;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/preprocessing/pipelines`, pipelineConfig);
    return response.data;
  }

  async monitorPipelineExecution(pipelineId: string): Promise<{
    status: 'running' | 'completed' | 'failed' | 'paused';
    progress: {
      current_step: string;
      completion_percentage: number;
      records_processed: number;
      estimated_remaining_time: number;
    };
    performance_metrics: {
      throughput_per_second: number;
      resource_utilization: any;
      error_rate: number;
    };
    step_details: Array<{
      step_id: string;
      status: string;
      duration: number;
      records_processed: number;
      errors: any[];
    }>;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/preprocessing/pipelines/${pipelineId}/status`);
    return response.data;
  }

  async validatePipelineOutput(pipelineId: string, validationRules: {
    data_quality_checks: string[];
    schema_validation: any;
    business_rules: any[];
    sample_validation: {
      sample_size: number;
      validation_criteria: any[];
    };
  }): Promise<{
    validation_result: {
      overall_status: 'pass' | 'fail' | 'warning';
      detailed_results: any[];
      recommendations: string[];
    };
    quality_report: any;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/preprocessing/pipelines/${pipelineId}/validate`, validationRules);
    return response.data;
  }

  // ========== 智能版本管理 ==========
  async createIntelligentVersioning(datasetName: string, versioningStrategy: {
    versioning_type: 'semantic' | 'timestamp' | 'content_hash' | 'custom';
    auto_versioning_rules: {
      trigger_conditions: any[];
      version_increment_logic: any;
    };
    metadata_tracking: {
      track_lineage: boolean;
      track_transformations: boolean;
      track_quality_metrics: boolean;
      custom_metadata: Record<string, any>;
    };
    retention_policy: {
      max_versions: number;
      retention_period: string;
      archival_strategy: string;
    };
  }): Promise<{
    versioning_config_id: string;
    next_version_preview: string;
    estimated_storage_impact: any;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/datasets/${datasetName}/intelligent-versioning`, versioningStrategy);
    return response.data;
  }

  async performSemanticVersionComparison(version1Id: string, version2Id: string, comparisonConfig: {
    comparison_depth: 'surface' | 'structural' | 'semantic' | 'comprehensive';
    focus_areas: string[];
    similarity_threshold: number;
  }): Promise<{
    comparison_summary: {
      overall_similarity: number;
      key_differences: any[];
      impact_assessment: any;
    };
    detailed_analysis: {
      schema_changes: any[];
      content_changes: any[];
      quality_changes: any[];
      statistical_differences: any[];
    };
    migration_recommendations: string[];
  }> {
    const response = await apiClient.post(`${this.baseUrl}/versions/${version1Id}/semantic-compare/${version2Id}`, comparisonConfig);
    return response.data;
  }

  async predictVersionImpact(versionId: string, targetEnvironments: string[]): Promise<{
    impact_analysis: Array<{
      environment: string;
      compatibility_score: number;
      potential_issues: any[];
      mitigation_strategies: string[];
    }>;
    rollback_plan: {
      steps: string[];
      estimated_downtime: number;
      risk_factors: string[];
    };
  }> {
    const response = await apiClient.post(`${this.baseUrl}/versions/${versionId}/impact-prediction`, { environments: targetEnvironments });
    return response.data;
  }

  // ========== 数据治理和合规性 ==========
  async establishDataGovernance(datasetId: string, governancePolicy: {
    data_classification: 'public' | 'internal' | 'confidential' | 'restricted';
    privacy_requirements: {
      contains_pii: boolean;
      anonymization_level: string;
      retention_requirements: any;
    };
    access_controls: {
      authorized_roles: string[];
      access_restrictions: any[];
      audit_requirements: any;
    };
    compliance_frameworks: string[];
  }): Promise<{
    governance_id: string;
    compliance_status: any;
    required_actions: string[];
  }> {
    const response = await apiClient.post(`${this.baseUrl}/datasets/${datasetId}/governance`, governancePolicy);
    return response.data;
  }

  async performComplianceAudit(datasetId: string, auditScope: {
    frameworks: string[];
    audit_depth: 'basic' | 'comprehensive' | 'detailed';
    include_historical: boolean;
  }): Promise<{
    audit_report: {
      overall_compliance_score: number;
      framework_scores: Record<string, number>;
      violations: any[];
      recommendations: string[];
    };
    remediation_plan: {
      priority_actions: any[];
      timeline_estimate: any;
      resource_requirements: any;
    };
  }> {
    const response = await apiClient.post(`${this.baseUrl}/datasets/${datasetId}/compliance-audit`, auditScope);
    return response.data;
  }

  // ========== 高级分析和洞察 ==========
  async generateDatasetInsights(datasetId: string, analysisConfig: {
    analysis_types: string[];
    statistical_depth: 'basic' | 'intermediate' | 'advanced';
    include_predictions: boolean;
    comparison_datasets?: string[];
  }): Promise<{
    insights_summary: {
      key_findings: string[];
      data_characteristics: any;
      quality_assessment: any;
      usage_patterns: any;
    };
    statistical_analysis: {
      descriptive_stats: any;
      correlation_analysis: any;
      distribution_analysis: any;
      anomaly_detection: any;
    };
    recommendations: {
      improvement_opportunities: string[];
      optimization_suggestions: string[];
      next_steps: string[];
    };
  }> {
    const response = await apiClient.post(`${this.baseUrl}/datasets/${datasetId}/insights`, analysisConfig);
    return response.data;
  }

  async performCrossDatasetAnalysis(datasetIds: string[], analysisType: {
    comparison_type: 'similarity' | 'complementarity' | 'redundancy' | 'gap_analysis';
    metrics: string[];
    correlation_analysis: boolean;
  }): Promise<{
    analysis_results: {
      dataset_relationships: any[];
      similarity_matrix: number[][];
      complementarity_score: any;
      recommendations: string[];
    };
    optimization_opportunities: any[];
  }> {
    const response = await apiClient.post(`${this.baseUrl}/cross-dataset-analysis`, { dataset_ids: datasetIds, ...analysisType });
    return response.data;
  }

  // ========== 自动化工作流管理 ==========
  async createAutomatedWorkflow(workflowConfig: {
    name: string;
    trigger_conditions: any[];
    workflow_steps: Array<{
      step_type: string;
      parameters: any;
      success_conditions: any[];
      failure_handling: any;
    }>;
    notification_settings: {
      channels: string[];
      recipients: string[];
      trigger_events: string[];
    };
    resource_limits: {
      max_execution_time: number;
      max_resources: any;
      priority_level: number;
    };
  }): Promise<{
    workflow_id: string;
    execution_schedule: any;
    estimated_resource_usage: any;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/workflows/automated`, workflowConfig);
    return response.data;
  }

  async monitorWorkflowPerformance(workflowId: string, timeRange?: {
    start_date: string;
    end_date: string;
  }): Promise<{
    performance_metrics: {
      execution_count: number;
      success_rate: number;
      average_duration: number;
      resource_utilization: any;
    };
    execution_history: Array<{
      execution_id: string;
      start_time: string;
      end_time: string;
      status: string;
      metrics: any;
    }>;
    optimization_suggestions: string[];
  }> {
    const params = timeRange ? `?start_date=${timeRange.start_date}&end_date=${timeRange.end_date}` : '';
    const response = await apiClient.get(`${this.baseUrl}/workflows/${workflowId}/performance${params}`);
    return response.data;
  }

  // ========== 协作和团队管理 ==========
  async setupTeamCollaboration(projectConfig: {
    project_name: string;
    team_members: Array<{
      user_id: string;
      role: string;
      permissions: string[];
    }>;
    collaboration_settings: {
      review_requirements: any;
      approval_workflows: any;
      communication_channels: any;
    };
    quality_standards: {
      minimum_quality_threshold: number;
      review_criteria: any[];
      escalation_rules: any[];
    };
  }): Promise<{
    project_id: string;
    collaboration_workspace: any;
    initial_assignments: any[];
  }> {
    const response = await apiClient.post(`${this.baseUrl}/collaboration/projects`, projectConfig);
    return response.data;
  }

  async getCollaborationMetrics(projectId: string): Promise<{
    team_performance: {
      overall_productivity: number;
      individual_contributions: any[];
      collaboration_efficiency: number;
    };
    quality_metrics: {
      review_turnaround_time: number;
      quality_consistency: number;
      issue_resolution_time: number;
    };
    project_progress: {
      milestone_completion: any[];
      timeline_adherence: number;
      resource_utilization: any;
    };
  }> {
    const response = await apiClient.get(`${this.baseUrl}/collaboration/projects/${projectId}/metrics`);
    return response.data;
  }
}

export const trainingDataServiceEnhanced = new TrainingDataServiceEnhanced();