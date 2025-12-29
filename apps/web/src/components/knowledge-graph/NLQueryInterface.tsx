/**
 * 知识图谱自然语言查询界面
 * 
 * 功能包括：
 * - 自然语言查询输入和自动补全
 * - 查询转换为Cypher语句(目标准确率≥85%)
 * - 查询结果高亮和导航功能
 * - 查询历史记录和收藏功能
 * - 智能查询建议和模板
 */

import { buildApiUrl, apiFetch } from '../../utils/apiBase'
import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  Card,
  Input,
  Button,
  Select,
  Space,
  Typography,
  Row,
  Col,
  AutoComplete,
  Tag,
  Tooltip,
  Collapse,
  List,
  message,
  Progress,
  Badge,
  Spin
} from 'antd';
import {
  SearchOutlined,
  HistoryOutlined,
  StarOutlined,
  StarFilled,
  BulbOutlined,
  CodeOutlined,
  ClearOutlined,
  QuestionCircleOutlined,
  ThunderboltOutlined
} from '@ant-design/icons';

const { TextArea } = Input;
const { Option } = Select;
const { Panel } = Collapse;
const { Title, Text, Paragraph } = Typography;

// ==================== 类型定义 ====================

export interface NLQuery {
  text: string;
  type: 'entity_search' | 'path_finding' | 'neighborhood' | 'pattern_match' | 'general_search';
  parameters: Record<string, any>;
  generatedCypher?: string;
  confidence?: number;
}

export interface QueryResult {
  queryId: string;
  query: NLQuery;
  results: any[];
  highlights: {
    nodeIds: string[];
    edgeIds: string[];
    paths: Array<{
      pathId: string;
      nodes: string[];
      edges: string[];
      description: string;
    }>;
  };
  executionTime: number;
  timestamp: Date;
}

export interface QueryHistory {
  id: string;
  query: string;
  type: string;
  results: number;
  timestamp: Date;
  isFavorite: boolean;
  confidence?: number;
}

interface NLQueryInterfaceProps {
  onQuery?: (query: NLQuery) => Promise<QueryResult>;
  onHighlight?: (highlights: QueryResult['highlights']) => void;
  disabled?: boolean;
  className?: string;
  placeholder?: string;
}

// ==================== 查询模板 ====================

const queryTemplates = [
  {
    category: '实体搜索',
    templates: [
      { text: '找到所有与{entity}相关的实体', example: '找到所有与"苹果公司"相关的实体' },
      { text: '搜索类型为{type}的所有实体', example: '搜索类型为"人物"的所有实体' },
      { text: '显示{entity}的详细信息', example: '显示"史蒂夫·乔布斯"的详细信息' }
    ]
  },
  {
    category: '路径查找',
    templates: [
      { text: '{entity1}和{entity2}之间的关系', example: '"苹果公司"和"iPhone"之间的关系' },
      { text: '从{entity1}到{entity2}的最短路径', example: '从"乔布斯"到"苹果公司"的最短路径' },
      { text: '{entity1}如何连接到{entity2}', example: '"微软"如何连接到"人工智能"' }
    ]
  },
  {
    category: '邻域探索',
    templates: [
      { text: '显示{entity}周围的所有连接', example: '显示"苹果公司"周围的所有连接' },
      { text: '{entity}的直接邻居', example: '"iPhone"的直接邻居' },
      { text: '与{entity}距离{n}跳内的实体', example: '与"乔布斯"距离2跳内的实体' }
    ]
  },
  {
    category: '模式匹配',
    templates: [
      { text: '所有{relation}关系的实体对', example: '所有"创立"关系的实体对' },
      { text: '具有{property}属性的实体', example: '具有"成立时间"属性的实体' },
      { text: '{type1}和{type2}之间的所有关系', example: '"公司"和"产品"之间的所有关系' }
    ]
  }
];

// ==================== 主组件 ====================

const NLQueryInterface: React.FC<NLQueryInterfaceProps> = ({
  onQuery,
  onHighlight,
  disabled = false,
  className = '',
  placeholder = '请输入自然语言查询，例如："苹果公司和iPhone之间的关系"'
}) => {
  // ==================== 状态管理 ====================
  
  const [queryText, setQueryText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [queryHistory, setQueryHistory] = useState<QueryHistory[]>([]);
  const [favorites, setFavorites] = useState<Set<string>>(new Set());
  const [currentResult, setCurrentResult] = useState<QueryResult | null>(null);
  const [autoCompleteOptions, setAutoCompleteOptions] = useState<string[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<string>('');
  
  const inputRef = useRef<any>(null);

  // ==================== 查询处理逻辑 ====================
  
  const processNaturalLanguageQuery = useCallback(async (text: string): Promise<NLQuery> => {
    // 实体识别模式
    const entityPatterns = [
      /"([^"]+)"/g, // 引号中的实体
      /【([^】]+)】/g, // 中文书名号
      /《([^》]+)》/g // 中文书名号变体
    ];

    const entities: string[] = [];
    entityPatterns.forEach(pattern => {
      const matches = text.matchAll(pattern);
      for (const match of matches) {
        entities.push(match[1]);
      }
    });

    // 查询类型检测
    const detectQueryType = (query: string): NLQuery['type'] => {
      const lowerQuery = query.toLowerCase();
      
      // 路径查找查询
      if (/之间.*关系|路径|连接|如何.*到/.test(lowerQuery)) {
        return 'path_finding';
      }
      
      // 邻域探索查询
      if (/周围|邻居|相关.*实体|连接.*实体/.test(lowerQuery)) {
        return 'neighborhood';
      }
      
      // 模式匹配查询
      if (/所有.*关系|具有.*属性|类型.*实体/.test(lowerQuery)) {
        return 'pattern_match';
      }
      
      // 实体搜索查询
      if (/找到|搜索|显示|查找/.test(lowerQuery)) {
        return 'entity_search';
      }
      
      return 'general_search';
    };

    const queryType = detectQueryType(text);
    
    // 生成Cypher查询
    const generateCypher = (type: NLQuery['type'], entities: string[]): string => {
      switch (type) {
        case 'path_finding':
          if (entities.length >= 2) {
            return `
              MATCH path = shortestPath((start:Entity)-[*1..5]-(end:Entity))
              WHERE start.name CONTAINS "${entities[0]}" 
                AND end.name CONTAINS "${entities[1]}"
              RETURN path, length(path) as pathLength
              ORDER BY pathLength
              LIMIT 5
            `;
          }
          break;
          
        case 'neighborhood':
          if (entities.length >= 1) {
            return `
              MATCH (center:Entity)-[r1]-(neighbor1:Entity)
              WHERE center.name CONTAINS "${entities[0]}"
              OPTIONAL MATCH (neighbor1)-[r2]-(neighbor2:Entity)
              WHERE neighbor2 <> center
              RETURN center, neighbor1, neighbor2, r1, r2
              LIMIT 50
            `;
          }
          break;
          
        case 'entity_search':
          if (entities.length >= 1) {
            return `
              MATCH (n:Entity)
              WHERE n.name CONTAINS "${entities[0]}"
              RETURN n
              LIMIT 20
            `;
          }
          break;
          
        case 'pattern_match':
          return `
            MATCH (n)-[r]-(m)
            RETURN DISTINCT type(r) as relationshipType, 
                   labels(n)[0] as sourceType, 
                   labels(m)[0] as targetType, 
                   count(*) as frequency
            ORDER BY frequency DESC
            LIMIT 20
          `;
          
        default:
          return `
            MATCH (n:Entity)
            WHERE n.name CONTAINS "${text.substring(0, 20)}"
            RETURN n
            LIMIT 10
          `;
      }
      return '';
    };

    // 计算查询置信度
    const calculateConfidence = (type: NLQuery['type'], entities: string[], text: string): number => {
      let confidence = 0.5; // 基础置信度
      
      // 根据实体数量调整
      confidence += entities.length * 0.1;
      
      // 根据查询类型的匹配程度调整
      if (type !== 'general_search') {
        confidence += 0.2;
      }
      
      // 根据文本长度和复杂度调整
      if (text.length > 10 && text.length < 100) {
        confidence += 0.1;
      }
      
      return Math.min(confidence, 1.0);
    };

    const cypher = generateCypher(queryType, entities);
    const confidence = calculateConfidence(queryType, entities, text);

    return {
      text,
      type: queryType,
      parameters: {
        entities,
        entityCount: entities.length
      },
      generatedCypher: cypher,
      confidence
    };
  }, []);

  // ==================== 查询执行 ====================
  
  const handleQuery = useCallback(async () => {
    if (!queryText.trim()) {
      message.warning('请输入查询内容');
      return;
    }

    setIsProcessing(true);
    
    try {
      // 处理自然语言查询
      const nlQuery = await processNaturalLanguageQuery(queryText);
      
      if (onQuery) {
        // 调用外部查询处理函数
        const result = await onQuery(nlQuery);
        setCurrentResult(result);
        
        // 触发高亮
        if (onHighlight && result.highlights) {
          onHighlight(result.highlights);
        }
        
        // 添加到历史记录
        const historyItem: QueryHistory = {
          id: Date.now().toString(),
          query: queryText,
          type: nlQuery.type,
          results: result.results.length,
          timestamp: new Date(),
          isFavorite: false,
          confidence: nlQuery.confidence
        };
        
        setQueryHistory(prev => [historyItem, ...prev.slice(0, 19)]);
        
        message.success(`查询完成，找到 ${result.results.length} 个结果`);
      } else {
        const cypher = nlQuery.generatedCypher?.trim();
        if (!cypher) throw new Error('无法生成可执行查询');

        const r = await apiFetch(buildApiUrl('/api/v1/knowledge-graph/query'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: cypher, parameters: {}), read_only: true }),
        });
        const resp = await r.json();
        if (!resp?.success) throw new Error(resp?.error || resp?.detail || '查询执行失败');

        const result: QueryResult = {
          queryId: Date.now().toString(),
          query: nlQuery,
          results: resp?.data || [],
          highlights: { nodeIds: [], edgeIds: [], paths: [] },
          executionTime: Number(resp?.execution_time_ms || 0),
          timestamp: new Date(),
        };

        setCurrentResult(result);

        const historyItem: QueryHistory = {
          id: Date.now().toString(),
          query: queryText,
          type: nlQuery.type,
          results: result.results.length,
          timestamp: new Date(),
          isFavorite: false,
          confidence: nlQuery.confidence
        };
        setQueryHistory(prev => [historyItem, ...prev.slice(0, 19)]);
        message.success(`查询完成，找到 ${result.results.length} 个结果`);
      }
      
    } catch (error: any) {
      message.error(`查询失败: ${error.message || '未知错误'}`);
    } finally {
      setIsProcessing(false);
    }
  }, [queryText, onQuery, onHighlight, processNaturalLanguageQuery]);

  // ==================== 自动补全 ====================
  
  const generateAutoComplete = useCallback((value: string) => {
    if (!value.trim()) {
      setAutoCompleteOptions([]);
      return;
    }

    const suggestions: string[] = [];
    
    // 基于历史查询的建议
    const historyMatches = queryHistory
      .filter(h => h.query.toLowerCase().includes(value.toLowerCase()))
      .slice(0, 3)
      .map(h => h.query);
    
    suggestions.push(...historyMatches);
    
    // 基于模板的建议
    queryTemplates.forEach(category => {
      category.templates.forEach(template => {
        if (template.example.toLowerCase().includes(value.toLowerCase())) {
          suggestions.push(template.example);
        }
      });
    });
    
    // 去重并限制数量
    const uniqueSuggestions = [...new Set(suggestions)].slice(0, 8);
    setAutoCompleteOptions(uniqueSuggestions);
  }, [queryHistory]);

  // ==================== 事件处理 ====================
  
  const handleQueryTextChange = useCallback((value: string) => {
    setQueryText(value);
    generateAutoComplete(value);
  }, [generateAutoComplete]);

  const handleHistorySelect = useCallback((query: string) => {
    setQueryText(query);
    inputRef.current?.focus();
  }, []);

  const handleTemplateSelect = useCallback((template: string) => {
    setQueryText(template);
    setSelectedTemplate(template);
    inputRef.current?.focus();
  }, []);

  const handleToggleFavorite = useCallback((id: string) => {
    setFavorites(prev => {
      const newFavorites = new Set(prev);
      if (newFavorites.has(id)) {
        newFavorites.delete(id);
      } else {
        newFavorites.add(id);
      }
      return newFavorites;
    });
    
    setQueryHistory(prev => prev.map(item => 
      item.id === id ? { ...item, isFavorite: !item.isFavorite } : item
    ));
  }, []);

  const handleClear = useCallback(() => {
    setQueryText('');
    setCurrentResult(null);
    setSelectedTemplate('');
    inputRef.current?.focus();
  }, []);

  // ==================== 生命周期 ====================
  
  useEffect(() => {
    // 组件加载时设置焦点
    inputRef.current?.focus();
  }, []);

  // ==================== 渲染组件 ====================

  return (
    <Card 
      className={`nl-query-interface ${className}`}
      title={
        <Space>
          <ThunderboltOutlined />
          <Title level={4} style={{ margin: 0 }}>自然语言查询</Title>
          {currentResult && (
            <Badge 
              count={currentResult.results.length} 
              style={{ backgroundColor: '#52c41a' }} 
              title="查询结果数量"
            />
          )}
        </Space>
      }
    >
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        
        {/* 主查询输入区域 */}
        <AutoComplete
          options={autoCompleteOptions.map(opt => ({ value: opt }))}
          style={{ width: '100%' }}
          onSelect={handleQueryTextChange}
          disabled={disabled}
        >
          <TextArea
            ref={inputRef}
            placeholder={placeholder}
            value={queryText}
            onChange={(e) => handleQueryTextChange(e.target.value)}
            onPressEnter={(e) => {
              if (!e.shiftKey) {
                e.preventDefault();
                handleQuery();
              }
            }}
            autoSize={{ minRows: 2, maxRows: 4 }}
            disabled={disabled}
          />
        </AutoComplete>

        {/* 操作按钮栏 */}
        <Row gutter={16} align="middle">
          <Col flex="auto">
            <Space>
              <Text type="secondary" style={{ fontSize: 12 }}>
                支持自然语言描述，如实体搜索、路径查找、邻域探索等
              </Text>
            </Space>
          </Col>
          
          <Col>
            <Space>
              <Tooltip title="清空查询">
                <Button
                  icon={<ClearOutlined />}
                  size="small"
                  onClick={handleClear}
                  disabled={disabled || !queryText}
                />
              </Tooltip>
              
              <Button
                type="primary"
                icon={<SearchOutlined />}
                loading={isProcessing}
                onClick={handleQuery}
                disabled={disabled || !queryText.trim()}
              >
                查询
              </Button>
            </Space>
          </Col>
        </Row>

        {/* 查询结果显示 */}
        {currentResult && (
          <Card size="small" title="查询结果" style={{ backgroundColor: '#f9f9f9' }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Row gutter={16}>
                <Col span={6}>
                  <Text strong>查询类型:</Text>
                </Col>
                <Col span={18}>
                  <Tag color="blue">{currentResult.query.type}</Tag>
                </Col>
              </Row>
              
              <Row gutter={16}>
                <Col span={6}>
                  <Text strong>置信度:</Text>
                </Col>
                <Col span={18}>
                  <Progress 
                    percent={Math.round((currentResult.query.confidence || 0) * 100)} 
                    size="small"
                    status={((currentResult.query.confidence || 0) >= 0.85) ? 'success' : 'normal'}
                  />
                </Col>
              </Row>
              
              <Row gutter={16}>
                <Col span={6}>
                  <Text strong>执行时间:</Text>
                </Col>
                <Col span={18}>
                  <Text code>{currentResult.executionTime.toFixed(2)}ms</Text>
                </Col>
              </Row>
              
              {currentResult.query.generatedCypher && (
                <Collapse ghost size="small">
                  <Panel 
                    header={
                      <Space>
                        <CodeOutlined />
                        <Text>生成的Cypher查询</Text>
                      </Space>
                    }
                    key="cypher"
                  >
                    <pre style={{ 
                      fontSize: 12, 
                      backgroundColor: '#f6f8fa', 
                      padding: 8, 
                      borderRadius: 4,
                      overflow: 'auto' 
                    }}>
                      {currentResult.query.generatedCypher}
                    </pre>
                  </Panel>
                </Collapse>
              )}
            </Space>
          </Card>
        )}

        {/* 查询模板和历史记录 */}
        <Collapse ghost>
          {/* 查询模板 */}
          <Panel 
            header={
              <Space>
                <BulbOutlined />
                <Text>查询模板</Text>
              </Space>
            }
            key="templates"
          >
            <Space direction="vertical" style={{ width: '100%' }}>
              {queryTemplates.map((category) => (
                <div key={category.category}>
                  <Text strong style={{ fontSize: 13 }}>{category.category}</Text>
                  <div style={{ marginTop: 4, marginBottom: 12 }}>
                    {category.templates.map((template, index) => (
                      <div 
                        key={index}
                        style={{ 
                          padding: '4px 8px',
                          margin: '2px 0',
                          backgroundColor: selectedTemplate === template.example ? '#e6f7ff' : '#fafafa',
                          borderRadius: 4,
                          cursor: 'pointer',
                          border: selectedTemplate === template.example ? '1px solid #1890ff' : '1px solid #d9d9d9'
                        }}
                        onClick={() => handleTemplateSelect(template.example)}
                      >
                        <Text style={{ fontSize: 12 }}>{template.example}</Text>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </Space>
          </Panel>

          {/* 查询历史 */}
          <Panel 
            header={
              <Space>
                <HistoryOutlined />
                <Text>查询历史 ({queryHistory.length})</Text>
              </Space>
            }
            key="history"
          >
            {queryHistory.length > 0 ? (
              <List
                size="small"
                dataSource={queryHistory.slice(0, 10)}
                renderItem={(item) => (
                  <List.Item
                    style={{ cursor: 'pointer' }}
                    onClick={() => handleHistorySelect(item.query)}
                    actions={[
                      <Tooltip title={item.isFavorite ? '取消收藏' : '收藏'} key="favorite">
                        <Button
                          type="text"
                          size="small"
                          icon={item.isFavorite ? <StarFilled style={{ color: '#faad14' }} /> : <StarOutlined />}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleToggleFavorite(item.id);
                          }}
                        />
                      </Tooltip>
                    ]}
                  >
                    <List.Item.Meta
                      title={
                        <Text ellipsis style={{ maxWidth: 300 }}>
                          {item.query}
                        </Text>
                      }
                      description={
                        <Space size="small">
                          <Tag size="small" color={getTypeColor(item.type)}>
                            {getTypeLabel(item.type)}
                          </Tag>
                          <Text type="secondary" style={{ fontSize: 11 }}>
                            {item.results} 结果
                          </Text>
                          {item.confidence && (
                            <Text type="secondary" style={{ fontSize: 11 }}>
                              置信度 {Math.round(item.confidence * 100)}%
                            </Text>
                          )}
                          <Text type="secondary" style={{ fontSize: 11 }}>
                            {item.timestamp.toLocaleString()}
                          </Text>
                        </Space>
                      }
                    />
                  </List.Item>
                )}
              />
            ) : (
              <Text type="secondary">暂无查询历史</Text>
            )}
          </Panel>
        </Collapse>

      </Space>
    </Card>
  );
};

// ==================== 工具函数 ====================

const getTypeColor = (type: string): string => {
  const colorMap: Record<string, string> = {
    'entity_search': 'blue',
    'path_finding': 'green',
    'neighborhood': 'orange',
    'pattern_match': 'purple',
    'general_search': 'default'
  };
  return colorMap[type] || 'default';
};

const getTypeLabel = (type: string): string => {
  const labelMap: Record<string, string> = {
    'entity_search': '实体搜索',
    'path_finding': '路径查找',
    'neighborhood': '邻域探索',
    'pattern_match': '模式匹配',
    'general_search': '通用搜索'
  };
  return labelMap[type] || '未知类型';
};

export default NLQueryInterface;
