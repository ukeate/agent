import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  CreditCard, 
  Building, 
  ShoppingCart, 
  UserCheck,
  TrendingUp,
  Shield,
  Briefcase,
  Heart,
  Play,
  Settings,
  Info,
  ChevronRight,
  Star,
  AlertTriangle,
  CheckCircle,
  Clock
} from 'lucide-react';

interface DemoScenario {
  id: string;
  title: string;
  category: 'finance' | 'healthcare' | 'ecommerce' | 'hr' | 'security';
  description: string;
  complexity: 'beginner' | 'intermediate' | 'advanced';
  duration: string;
  features: string[];
  icon: React.ReactNode;
  previewData: {
    decision_outcome: string;
    confidence: number;
    key_factors: string[];
    risk_level: 'low' | 'medium' | 'high';
  };
  datasets: {
    name: string;
    description: string;
    records: number;
  }[];
  learningObjectives: string[];
}

interface DemoScenarioSelectorProps {
  onSelectScenario: (scenario: DemoScenario) => void;
  selectedScenario?: string;
  className?: string;
}

const DemoScenarioSelector: React.FC<DemoScenarioSelectorProps> = ({
  onSelectScenario,
  selectedScenario,
  className = ''
}) => {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState<string | null>(null);

  const scenarios: DemoScenario[] = [
    {
      id: 'loan-approval',
      title: '银行贷款审批决策',
      category: 'finance',
      description: '分析个人贷款申请的风险评估和决策过程，展示信用评分、收入验证、风险因子等关键要素。',
      complexity: 'intermediate',
      duration: '15-20分钟',
      features: ['置信度分析', '风险评估', '反事实分析', 'CoT推理'],
      icon: <CreditCard className="h-6 w-6" />,
      previewData: {
        decision_outcome: '贷款申请批准',
        confidence: 0.85,
        key_factors: ['信用评分: 750', '月收入: ¥15,000', '工作年限: 5年'],
        risk_level: 'low'
      },
      datasets: [
        { name: '信用评分数据', description: '央行征信系统数据', records: 10000 },
        { name: '收入验证记录', description: '银行流水和工资单', records: 5000 },
        { name: '历史违约记录', description: '过往贷款违约统计', records: 2000 }
      ],
      learningObjectives: [
        '理解金融风险评估的多维度分析',
        '掌握置信度计算在信贷决策中的应用',
        '学习反事实分析在风险管理中的价值',
        '体验可解释AI在金融合规中的重要性'
      ]
    },
    {
      id: 'medical-diagnosis',
      title: '医疗诊断辅助系统',
      category: 'healthcare',
      description: '基于症状、检查结果和病史的疾病诊断建议，重点展示医疗AI的决策透明度和可信度。',
      complexity: 'advanced',
      duration: '25-30分钟',
      features: ['症状权重分析', '检查结果解读', '诊断置信度', '医疗知识推理'],
      icon: <Heart className="h-6 w-6" />,
      previewData: {
        decision_outcome: '疑似冠心病，建议进一步检查',
        confidence: 0.72,
        key_factors: ['胸痛症状', '心电图异常', '家族病史阳性'],
        risk_level: 'medium'
      },
      datasets: [
        { name: '症状数据库', description: '临床症状特征库', records: 50000 },
        { name: '检查结果集', description: '各类医学检查数据', records: 20000 },
        { name: '疾病知识图谱', description: '医学知识关联网络', records: 100000 }
      ],
      learningObjectives: [
        '了解医疗AI的决策支持机制',
        '学习症状与疾病的关联分析',
        '掌握医疗决策的不确定性量化',
        '理解医疗AI的伦理和安全考量'
      ]
    },
    {
      id: 'product-recommendation',
      title: '电商个性化推荐',
      category: 'ecommerce',
      description: '分析用户行为、偏好和商品特征，生成个性化推荐并解释推荐理由。',
      complexity: 'beginner',
      duration: '10-15分钟',
      features: ['用户画像分析', '商品匹配度', '行为序列分析', '推荐解释'],
      icon: <ShoppingCart className="h-6 w-6" />,
      previewData: {
        decision_outcome: '推荐iPhone 15 Pro',
        confidence: 0.91,
        key_factors: ['浏览苹果产品', '高端手机偏好', '技术爱好者'],
        risk_level: 'low'
      },
      datasets: [
        { name: '用户行为数据', description: '用户浏览购买记录', records: 100000 },
        { name: '商品特征库', description: '商品属性和标签', records: 50000 },
        { name: '评价和反馈', description: '用户评价情感分析', records: 200000 }
      ],
      learningObjectives: [
        '理解推荐系统的协同过滤机制',
        '学习用户画像和商品画像的构建',
        '掌握推荐结果的可解释性设计',
        '体验个性化算法的透明度要求'
      ]
    },
    {
      id: 'employee-evaluation',
      title: '员工绩效评估',
      category: 'hr',
      description: '综合员工工作表现、技能发展和团队协作，提供客观的绩效评估和发展建议。',
      complexity: 'intermediate',
      duration: '20-25分钟',
      features: ['多维度评估', '同事反馈分析', '技能差距识别', '发展路径建议'],
      icon: <Briefcase className="h-6 w-6" />,
      previewData: {
        decision_outcome: '绩效评级: 优秀 (A)',
        confidence: 0.88,
        key_factors: ['项目完成质量高', '团队协作优秀', '学习能力强'],
        risk_level: 'low'
      },
      datasets: [
        { name: '工作任务记录', description: '项目完成情况统计', records: 15000 },
        { name: '360度反馈', description: '上级下级同事评价', records: 8000 },
        { name: '技能评估数据', description: '专业技能测试结果', records: 12000 }
      ],
      learningObjectives: [
        '了解HR Analytics的数据驱动决策',
        '学习多维度员工评估体系',
        '掌握绩效评估的公平性和透明度',
        '理解人力资源AI的伦理考量'
      ]
    },
    {
      id: 'fraud-detection',
      title: '金融反欺诈检测',
      category: 'security',
      description: '实时监控交易行为，识别可疑模式并评估欺诈风险，重点展示安全AI的决策逻辑。',
      complexity: 'advanced',
      duration: '20-25分钟',
      features: ['异常模式识别', '风险评分计算', '实时决策解释', '误报分析'],
      icon: <Shield className="h-6 w-6" />,
      previewData: {
        decision_outcome: '检测到高风险交易',
        confidence: 0.94,
        key_factors: ['异地大额交易', '非正常时间', '设备指纹异常'],
        risk_level: 'high'
      },
      datasets: [
        { name: '交易行为数据', description: '历史交易模式分析', records: 500000 },
        { name: '设备指纹库', description: '用户设备特征记录', records: 100000 },
        { name: '欺诈案例库', description: '已确认的欺诈交易', records: 5000 }
      ],
      learningObjectives: [
        '理解异常检测算法的工作原理',
        '学习安全AI的实时决策机制',
        '掌握误报与漏报的平衡策略',
        '了解安全AI的可解释性要求'
      ]
    },
    {
      id: 'supply-chain',
      title: '供应链风险管理',
      category: 'finance',
      description: '分析供应商财务状况、地缘政治风险和市场波动，预测供应链中断风险。',
      complexity: 'advanced',
      duration: '30-35分钟',
      features: ['多源数据融合', '风险传播分析', '情景模拟', '决策树解析'],
      icon: <Building className="h-6 w-6" />,
      previewData: {
        decision_outcome: '建议寻找备用供应商',
        confidence: 0.76,
        key_factors: ['地缘政治紧张', '供应商财务恶化', '运输成本上升'],
        risk_level: 'high'
      },
      datasets: [
        { name: '供应商数据', description: '财务和运营状况', records: 5000 },
        { name: '地缘政治指标', description: '政治风险评估数据', records: 200 },
        { name: '市场价格数据', description: '原材料价格波动', records: 50000 }
      ],
      learningObjectives: [
        '了解复杂系统的风险建模',
        '学习多源异构数据的融合分析',
        '掌握供应链风险的量化评估',
        '理解商业智能中的可解释性'
      ]
    }
  ];

  const categories = [
    { id: 'finance', name: '金融服务', icon: <CreditCard className="h-5 w-5" />, color: 'blue' },
    { id: 'healthcare', name: '医疗健康', icon: <Heart className="h-5 w-5" />, color: 'red' },
    { id: 'ecommerce', name: '电商零售', icon: <ShoppingCart className="h-5 w-5" />, color: 'green' },
    { id: 'hr', name: '人力资源', icon: <Briefcase className="h-5 w-5" />, color: 'purple' },
    { id: 'security', name: '安全防护', icon: <Shield className="h-5 w-5" />, color: 'orange' }
  ];

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'beginner':
        return 'bg-green-100 text-green-700 border-green-200';
      case 'intermediate':
        return 'bg-yellow-100 text-yellow-700 border-yellow-200';
      case 'advanced':
        return 'bg-red-100 text-red-700 border-red-200';
      default:
        return 'bg-gray-100 text-gray-700 border-gray-200';
    }
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low':
        return 'text-green-600';
      case 'medium':
        return 'text-yellow-600';
      case 'high':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  const getCategoryColor = (category: string) => {
    const categoryConfig = categories.find(c => c.id === category);
    const color = categoryConfig?.color || 'gray';
    return `bg-${color}-100 text-${color}-700 border-${color}-200`;
  };

  const filteredScenarios = selectedCategory 
    ? scenarios.filter(s => s.category === selectedCategory)
    : scenarios;

  return (
    <div className={`space-y-6 ${className}`}>
      {/* 标题和说明 */}
      <div className="text-center">
        <h2 className="text-2xl font-bold mb-2">演示场景选择器</h2>
        <p className="text-gray-600 max-w-2xl mx-auto">
          选择不同的业务场景来体验解释性AI在各个领域的应用。每个场景都包含真实的业务逻辑和完整的决策解释过程。
        </p>
      </div>

      {/* 分类过滤器 */}
      <div className="flex flex-wrap justify-center gap-3">
        <Button
          onClick={() => setSelectedCategory(null)}
          variant={selectedCategory === null ? "default" : "outline"}
          className="flex items-center space-x-2"
        >
          <span>全部场景</span>
          <Badge variant="secondary">{scenarios.length}</Badge>
        </Button>
        {categories.map(category => {
          const count = scenarios.filter(s => s.category === category.id).length;
          return (
            <Button
              key={category.id}
              onClick={() => setSelectedCategory(category.id)}
              variant={selectedCategory === category.id ? "default" : "outline"}
              className="flex items-center space-x-2"
            >
              {category.icon}
              <span>{category.name}</span>
              <Badge variant="secondary">{count}</Badge>
            </Button>
          );
        })}
      </div>

      {/* 场景卡片网格 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredScenarios.map(scenario => {
          const isSelected = selectedScenario === scenario.id;
          const isExpanded = showDetails === scenario.id;
          
          return (
            <Card 
              key={scenario.id}
              className={`transition-all duration-200 ${
                isSelected ? 'ring-2 ring-blue-500 border-blue-200' : 'hover:shadow-lg'
              } ${isExpanded ? 'md:col-span-2 lg:col-span-3' : ''}`}
            >
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${getCategoryColor(scenario.category)}`}>
                      {scenario.icon}
                    </div>
                    <div>
                      <CardTitle className="text-lg">{scenario.title}</CardTitle>
                      <div className="flex items-center space-x-2 mt-1">
                        <Badge className={getComplexityColor(scenario.complexity)} variant="outline">
                          {scenario.complexity === 'beginner' ? '初级' :
                           scenario.complexity === 'intermediate' ? '中级' : '高级'}
                        </Badge>
                        <div className="flex items-center text-xs text-gray-500">
                          <Clock className="h-3 w-3 mr-1" />
                          {scenario.duration}
                        </div>
                      </div>
                    </div>
                  </div>
                  <Button
                    onClick={() => setShowDetails(isExpanded ? null : scenario.id)}
                    variant="ghost"
                    size="sm"
                  >
                    <Info className="h-4 w-4" />
                  </Button>
                </div>
              </CardHeader>

              <CardContent className="space-y-4">
                <p className="text-sm text-gray-600">{scenario.description}</p>

                {/* 预览数据 */}
                <div className="bg-gray-50 p-3 rounded-lg">
                  <h4 className="font-medium text-sm mb-2">决策预览</h4>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span>结果:</span>
                      <span className="font-medium">{scenario.previewData.decision_outcome}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>置信度:</span>
                      <span className="font-medium">{(scenario.previewData.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>风险等级:</span>
                      <span className={`font-medium ${getRiskColor(scenario.previewData.risk_level)}`}>
                        {scenario.previewData.risk_level === 'low' ? '低' :
                         scenario.previewData.risk_level === 'medium' ? '中' : '高'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* 功能特性 */}
                <div>
                  <h4 className="font-medium text-sm mb-2">包含功能</h4>
                  <div className="flex flex-wrap gap-1">
                    {scenario.features.slice(0, 3).map((feature, i) => (
                      <Badge key={i} variant="outline" className="text-xs">
                        {feature}
                      </Badge>
                    ))}
                    {scenario.features.length > 3 && (
                      <Badge variant="outline" className="text-xs">
                        +{scenario.features.length - 3}
                      </Badge>
                    )}
                  </div>
                </div>

                {/* 详细信息 */}
                {isExpanded && (
                  <div className="space-y-4 border-t pt-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* 数据集信息 */}
                      <div>
                        <h4 className="font-medium mb-3 flex items-center">
                          <Database className="h-4 w-4 mr-2" />
                          数据集信息
                        </h4>
                        <div className="space-y-2">
                          {scenario.datasets.map((dataset, i) => (
                            <div key={i} className="border rounded p-3">
                              <div className="flex justify-between items-start mb-1">
                                <h5 className="font-medium text-sm">{dataset.name}</h5>
                                <Badge variant="outline" className="text-xs">
                                  {dataset.records.toLocaleString()} 条
                                </Badge>
                              </div>
                              <p className="text-xs text-gray-600">{dataset.description}</p>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* 学习目标 */}
                      <div>
                        <h4 className="font-medium mb-3 flex items-center">
                          <Star className="h-4 w-4 mr-2" />
                          学习目标
                        </h4>
                        <div className="space-y-2">
                          {scenario.learningObjectives.map((objective, i) => (
                            <div key={i} className="flex items-start space-x-2 text-sm">
                              <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 flex-shrink-0" />
                              <span className="text-gray-700">{objective}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>

                    {/* 关键因素预览 */}
                    <div>
                      <h4 className="font-medium mb-3">关键决策因素</h4>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        {scenario.previewData.key_factors.map((factor, i) => (
                          <div key={i} className="bg-blue-50 p-3 rounded border-l-4 border-blue-500">
                            <div className="text-sm font-medium text-blue-800">{factor}</div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* 完整功能列表 */}
                    <div>
                      <h4 className="font-medium mb-3">完整功能列表</h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                        {scenario.features.map((feature, i) => (
                          <div key={i} className="flex items-center space-x-2 text-sm">
                            <CheckCircle className="h-3 w-3 text-green-600" />
                            <span>{feature}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* 操作按钮 */}
                <div className="flex justify-between items-center pt-3 border-t">
                  <div className="flex items-center space-x-2">
                    <TrendingUp className={`h-4 w-4 ${getRiskColor(scenario.previewData.risk_level)}`} />
                    <span className="text-xs text-gray-600">
                      复杂度: {scenario.complexity === 'beginner' ? '初级' :
                               scenario.complexity === 'intermediate' ? '中级' : '高级'}
                    </span>
                  </div>
                  <Button
                    onClick={() => onSelectScenario(scenario)}
                    className="flex items-center space-x-2"
                    disabled={isSelected}
                  >
                    <Play className="h-4 w-4" />
                    <span>{isSelected ? '已选择' : '开始体验'}</span>
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* 底部说明 */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-start space-x-3">
          <Info className="h-5 w-5 text-blue-600 mt-0.5" />
          <div>
            <h3 className="font-medium text-blue-800 mb-1">使用说明</h3>
            <div className="text-sm text-blue-700 space-y-1">
              <p>• 每个场景都包含完整的数据处理和决策解释流程</p>
              <p>• 建议按照从初级到高级的顺序体验不同复杂度的场景</p>
              <p>• 每个场景都有对应的学习目标，帮助你掌握特定领域的AI应用</p>
              <p>• 可以重复体验同一场景来深入理解决策逻辑和技术实现</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DemoScenarioSelector;