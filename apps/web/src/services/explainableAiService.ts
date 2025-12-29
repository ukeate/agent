import apiClient from './apiClient';

export interface ExplanationComponent {
  factor_name: string;
  factor_value: any;
  weight: number;
  impact_score: number;
  evidence_type: string;
  evidence_source: string;
  evidence_content: string;
}

export interface ConfidenceMetrics {
  overall_confidence: number;
  uncertainty_score: number;
  confidence_interval_lower?: number;
  confidence_interval_upper?: number;
  confidence_sources?: string[];
}

export interface CounterfactualScenario {
  scenario_name: string;
  predicted_outcome: string;
  probability: number;
  impact_difference: number;
  explanation: string;
}

export interface DecisionExplanation {
  id: string;
  decision_id: string;
  explanation_type: string;
  explanation_level: string;
  decision_description: string;
  decision_outcome: string;
  summary_explanation: string;
  detailed_explanation?: string;
  components: ExplanationComponent[];
  confidence_metrics: ConfidenceMetrics;
  counterfactuals: CounterfactualScenario[];
  generated_at: string;
  metadata?: Record<string, any>;
}

export interface ExplanationRequest {
  decision_id: string;
  decision_context: string;
  explanation_type?: 'decision' | 'reasoning' | 'workflow';
  explanation_level?: 'summary' | 'detailed' | 'technical';
  style?: string;
  factors?: Array<{
    factor_name: string;
    factor_value: any;
    weight: number;
    impact: number;
    source: string;
  }>;
  use_cot_reasoning?: boolean;
  reasoning_mode?: string;
}

export interface CoTReasoningRequest {
  decision_id: string;
  decision_context: string;
  reasoning_mode?: 'analytical' | 'deductive' | 'inductive' | 'abductive';
  explanation_level?: 'summary' | 'detailed' | 'technical';
  factors?: Array<{
    factor_name: string;
    factor_value: any;
    weight: number;
    impact: number;
    source: string;
  }>;
}

export interface WorkflowExecutionRequest {
  workflow_id: string;
  workflow_name: string;
  nodes: Array<{
    node_id: string;
    node_type: string;
    node_name: string;
    input_data: Record<string, any>;
    output_data: Record<string, any>;
    execution_time: number;
    status: string;
    metadata?: Record<string, any>;
  }>;
  explanation_level?: 'summary' | 'detailed' | 'technical';
}

export interface DemoScenarioRequest {
  scenario_type: 'loan_approval' | 'medical_diagnosis' | 'investment_recommendation';
  complexity?: 'simple' | 'medium' | 'complex';
  include_cot?: boolean;
}

export interface ExplanationFormatRequest {
  explanation_id: string;
  output_format: 'html' | 'markdown' | 'json' | 'text' | 'xml';
  template_name?: string;
}

export interface ExplanationTypes {
  explanation_types: Array<{
    value: string;
    label: string;
    description: string;
  }>;
  explanation_levels: Array<{
    value: string;
    label: string;
    description: string;
  }>;
  reasoning_modes: Array<{
    value: string;
    label: string;
    description: string;
  }>;
  output_formats: Array<{
    value: string;
    label: string;
    description: string;
  }>;
}

export interface DemoScenarios {
  scenarios: Array<{
    type: string;
    name: string;
    description: string;
    complexity_levels: string[];
  }>;
}

class ExplainableAiService {
  private baseUrl = '/explainable-ai';

  // 生成解释
  async generateExplanation(request: ExplanationRequest): Promise<DecisionExplanation> {
    const response = await apiClient.post(`${this.baseUrl}/generate-explanation`, request);
    return response.data;
  }

  // 生成CoT推理解释  
  async generateCoTReasoning(request: CoTReasoningRequest): Promise<DecisionExplanation> {
    const response = await apiClient.post(`${this.baseUrl}/cot-reasoning`, request);
    return response.data;
  }

  // 生成工作流解释
  async generateWorkflowExplanation(request: WorkflowExecutionRequest): Promise<DecisionExplanation> {
    const response = await apiClient.post(`${this.baseUrl}/workflow-explanation`, request);
    return response.data;
  }

  // 格式化解释
  async formatExplanation(request: ExplanationFormatRequest): Promise<{
    explanation_id: string;
    format: string;
    content: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/format-explanation`, request);
    return response.data;
  }

  // 生成演示场景
  async generateDemoScenario(request: DemoScenarioRequest): Promise<DecisionExplanation> {
    const response = await apiClient.post(`${this.baseUrl}/demo-scenario`, request);
    return response.data;
  }

  // 获取解释类型
  async getExplanationTypes(): Promise<ExplanationTypes> {
    const response = await apiClient.get(`${this.baseUrl}/explanation-types`);
    return response.data;
  }

  // 获取演示场景
  async getDemoScenarios(): Promise<DemoScenarios> {
    const response = await apiClient.get(`${this.baseUrl}/demo-scenarios`);
    return response.data;
  }

  // 健康检查
  async healthCheck(): Promise<{
    status: string;
    service: string;
    timestamp: string;
    version: string;
    components: Record<string, string>;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/health`);
    return response.data;
  }

  // 预定义场景生成器
  async generateLoanApprovalExplanation(
    creditScore: number,
    annualIncome: number,
    employmentYears: number,
    debtRatio: number
  ): Promise<DecisionExplanation> {
    const request: ExplanationRequest = {
      decision_id: `loan_${Date.now()}`,
      decision_context: '贷款申请决策分析',
      explanation_type: 'decision',
      explanation_level: 'detailed',
      factors: [
        {
          factor_name: 'credit_score',
          factor_value: creditScore,
          weight: 0.35,
          impact: creditScore > 700 ? 0.8 : creditScore > 600 ? 0.6 : 0.3,
          source: 'credit_bureau'
        },
        {
          factor_name: 'annual_income', 
          factor_value: annualIncome,
          weight: 0.25,
          impact: annualIncome > 60000 ? 0.7 : annualIncome > 40000 ? 0.5 : 0.3,
          source: 'payroll'
        },
        {
          factor_name: 'employment_duration',
          factor_value: employmentYears,
          weight: 0.2,
          impact: employmentYears > 3 ? 0.6 : employmentYears > 1 ? 0.4 : 0.2,
          source: 'hr_system'
        },
        {
          factor_name: 'debt_ratio',
          factor_value: debtRatio,
          weight: 0.2,
          impact: debtRatio < 0.3 ? 0.5 : debtRatio < 0.5 ? 0.3 : 0.1,
          source: 'financial_calc'
        }
      ],
      use_cot_reasoning: true,
      reasoning_mode: 'analytical'
    };

    return this.generateExplanation(request);
  }

  // 医疗诊断解释
  async generateMedicalDiagnosisExplanation(
    symptomSeverity: string,
    labResults: string,
    patientHistory: string,
    imagingFindings: string
  ): Promise<DecisionExplanation> {
    const request: ExplanationRequest = {
      decision_id: `medical_${Date.now()}`,
      decision_context: '医疗诊断辅助决策',
      explanation_type: 'decision',
      explanation_level: 'detailed',
      factors: [
        {
          factor_name: 'symptom_severity',
          factor_value: symptomSeverity,
          weight: 0.4,
          impact: symptomSeverity === 'severe' ? 0.9 : symptomSeverity === 'moderate' ? 0.7 : 0.4,
          source: 'clinical_assessment'
        },
        {
          factor_name: 'lab_results',
          factor_value: labResults,
          weight: 0.3,
          impact: labResults === 'abnormal' ? 0.8 : labResults === 'borderline' ? 0.5 : 0.2,
          source: 'laboratory'
        },
        {
          factor_name: 'patient_history',
          factor_value: patientHistory,
          weight: 0.2,
          impact: patientHistory === 'relevant' ? 0.6 : patientHistory === 'partial' ? 0.4 : 0.1,
          source: 'medical_records'
        },
        {
          factor_name: 'imaging_findings',
          factor_value: imagingFindings,
          weight: 0.1,
          impact: imagingFindings === 'positive' ? 0.9 : imagingFindings === 'inconclusive' ? 0.3 : 0.1,
          source: 'radiology'
        }
      ],
      use_cot_reasoning: true,
      reasoning_mode: 'deductive'
    };

    return this.generateExplanation(request);
  }
}

export default new ExplainableAiService();