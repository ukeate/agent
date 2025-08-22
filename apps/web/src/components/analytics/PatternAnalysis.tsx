import React, { useState } from 'react';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { Progress } from '../ui/Progress';

interface Pattern {
  pattern_id: string;
  sequence: string[];
  support: number;
  confidence: number;
  frequency: number;
  user_count: number;
  description?: string;
}

interface PatternAnalysisProps {
  patterns: Pattern[];
}

export const PatternAnalysis: React.FC<PatternAnalysisProps> = ({ patterns }) => {
  const [selectedPattern, setSelectedPattern] = useState<Pattern | null>(null);
  const [sortBy, setSortBy] = useState<'support' | 'confidence' | 'frequency'>('support');
  const [filterMinSupport, setFilterMinSupport] = useState<number>(0.1);

  // 排序和过滤模式
  const processedPatterns = patterns
    .filter(pattern => pattern.support >= filterMinSupport)
    .sort((a, b) => b[sortBy] - a[sortBy]);

  // 获取支持度颜色
  const getSupportColor = (support: number) => {
    if (support >= 0.7) return 'bg-green-500';
    if (support >= 0.4) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  // 获取置信度颜色
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  // 格式化序列显示
  const formatSequence = (sequence: string[]) => {
    return sequence.map((step, index) => (
      <span key={index} className="inline-flex items-center">
        <Badge variant="outline" className="text-xs">
          {step}
        </Badge>
        {index < sequence.length - 1 && (
          <span className="mx-2 text-gray-400">→</span>
        )}
      </span>
    ));
  };

  // 计算模式强度
  const getPatternStrength = (pattern: Pattern) => {
    return (pattern.support * 0.4 + pattern.confidence * 0.4 + (pattern.frequency / 1000) * 0.2) * 100;
  };

  return (
    <div className="space-y-6">
      {/* 控制面板 */}
      <Card className="p-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">
              行为模式分析
            </h3>
            <p className="text-sm text-gray-600">
              发现用户行为序列中的频繁模式和规律
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* 排序选择 */}
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-600">排序：</span>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as any)}
                className="text-sm border border-gray-300 rounded px-2 py-1"
              >
                <option value="support">支持度</option>
                <option value="confidence">置信度</option>
                <option value="frequency">频率</option>
              </select>
            </div>

            {/* 最小支持度过滤 */}
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-600">最小支持度：</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={filterMinSupport}
                onChange={(e) => setFilterMinSupport(parseFloat(e.target.value))}
                className="w-20"
              />
              <span className="text-sm text-gray-600 w-8">
                {(filterMinSupport * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>
      </Card>

      {/* 模式概览统计 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-blue-600">
              {processedPatterns.length}
            </p>
            <p className="text-sm text-gray-600">发现的模式</p>
          </div>
        </Card>
        
        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-green-600">
              {processedPatterns.filter(p => p.support >= 0.5).length}
            </p>
            <p className="text-sm text-gray-600">高支持度模式</p>
          </div>
        </Card>
        
        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-purple-600">
              {processedPatterns.filter(p => p.confidence >= 0.8).length}
            </p>
            <p className="text-sm text-gray-600">高置信度模式</p>
          </div>
        </Card>
        
        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-orange-600">
              {Math.max(...processedPatterns.map(p => p.sequence.length), 0)}
            </p>
            <p className="text-sm text-gray-600">最长序列长度</p>
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 模式列表 */}
        <Card className="p-6">
          <h4 className="font-semibold text-gray-900 mb-4">
            频繁行为模式 ({processedPatterns.length})
          </h4>
          
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {processedPatterns.map((pattern) => (
              <div
                key={pattern.pattern_id}
                className={`p-4 border rounded-md cursor-pointer transition-colors ${
                  selectedPattern?.pattern_id === pattern.pattern_id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => setSelectedPattern(pattern)}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      {formatSequence(pattern.sequence)}
                    </div>
                    {pattern.description && (
                      <p className="text-xs text-gray-600 mb-2">
                        {pattern.description}
                      </p>
                    )}
                  </div>
                  <div className="flex items-center space-x-2 ml-4">
                    <div
                      className={`w-3 h-3 rounded-full ${getSupportColor(pattern.support)}`}
                      title={`支持度: ${(pattern.support * 100).toFixed(1)}%`}
                    />
                  </div>
                </div>
                
                <div className="grid grid-cols-3 gap-4 text-xs">
                  <div>
                    <span className="text-gray-500">支持度</span>
                    <p className="font-medium">
                      {(pattern.support * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <span className="text-gray-500">置信度</span>
                    <p className={`font-medium ${getConfidenceColor(pattern.confidence)}`}>
                      {(pattern.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <span className="text-gray-500">频率</span>
                    <p className="font-medium">
                      {pattern.frequency}
                    </p>
                  </div>
                </div>
                
                {/* 模式强度 */}
                <div className="mt-3">
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="text-gray-500">模式强度</span>
                    <span>{getPatternStrength(pattern).toFixed(1)}%</span>
                  </div>
                  <Progress 
                    value={getPatternStrength(pattern)} 
                    max={100}
                    className="h-2"
                  />
                </div>
              </div>
            ))}
            
            {processedPatterns.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                <p>没有找到符合条件的行为模式</p>
                <p className="text-sm mt-1">尝试降低最小支持度阈值</p>
              </div>
            )}
          </div>
        </Card>

        {/* 模式详情 */}
        <Card className="p-6">
          <h4 className="font-semibold text-gray-900 mb-4">
            模式详情分析
          </h4>
          
          {selectedPattern ? (
            <div className="space-y-6">
              {/* 序列可视化 */}
              <div>
                <h5 className="text-sm font-medium text-gray-700 mb-3">行为序列</h5>
                <div className="bg-gray-50 p-4 rounded-md">
                  <div className="flex items-center justify-center space-x-2 flex-wrap">
                    {selectedPattern.sequence.map((step, index) => (
                      <React.Fragment key={index}>
                        <div className="flex flex-col items-center">
                          <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-bold">
                            {index + 1}
                          </div>
                          <Badge className="mt-2 text-xs">
                            {step}
                          </Badge>
                        </div>
                        {index < selectedPattern.sequence.length - 1 && (
                          <div className="flex items-center">
                            <div className="w-8 h-0.5 bg-gray-300"></div>
                            <span className="text-gray-400 text-lg">▶</span>
                          </div>
                        )}
                      </React.Fragment>
                    ))}
                  </div>
                </div>
              </div>

              {/* 统计指标 */}
              <div>
                <h5 className="text-sm font-medium text-gray-700 mb-3">关键指标</h5>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-blue-50 p-3 rounded-md">
                    <p className="text-xs text-blue-600 mb-1">支持度</p>
                    <p className="text-lg font-bold text-blue-900">
                      {(selectedPattern.support * 100).toFixed(1)}%
                    </p>
                    <Progress 
                      value={selectedPattern.support * 100}
                      max={100}
                      className="h-2 mt-2"
                    />
                  </div>
                  
                  <div className="bg-green-50 p-3 rounded-md">
                    <p className="text-xs text-green-600 mb-1">置信度</p>
                    <p className="text-lg font-bold text-green-900">
                      {(selectedPattern.confidence * 100).toFixed(1)}%
                    </p>
                    <Progress 
                      value={selectedPattern.confidence * 100}
                      max={100}
                      className="h-2 mt-2"
                    />
                  </div>
                  
                  <div className="bg-purple-50 p-3 rounded-md">
                    <p className="text-xs text-purple-600 mb-1">出现频率</p>
                    <p className="text-lg font-bold text-purple-900">
                      {selectedPattern.frequency}
                    </p>
                    <p className="text-xs text-purple-600 mt-1">次</p>
                  </div>
                  
                  <div className="bg-orange-50 p-3 rounded-md">
                    <p className="text-xs text-orange-600 mb-1">涉及用户</p>
                    <p className="text-lg font-bold text-orange-900">
                      {selectedPattern.user_count}
                    </p>
                    <p className="text-xs text-orange-600 mt-1">人</p>
                  </div>
                </div>
              </div>

              {/* 业务洞察 */}
              <div>
                <h5 className="text-sm font-medium text-gray-700 mb-3">业务洞察</h5>
                <div className="space-y-2">
                  <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-md">
                    <p className="text-sm text-yellow-800">
                      <span className="font-medium">关键发现：</span>
                      该行为模式在 {selectedPattern.user_count} 个用户中出现了 {selectedPattern.frequency} 次，
                      显示出 {(selectedPattern.support * 100).toFixed(1)}% 的支持度。
                    </p>
                  </div>
                  
                  {selectedPattern.confidence >= 0.8 && (
                    <div className="p-3 bg-green-50 border border-green-200 rounded-md">
                      <p className="text-sm text-green-800">
                        <span className="font-medium">高置信度模式：</span>
                        {(selectedPattern.confidence * 100).toFixed(1)}% 的置信度表明这是一个可靠的行为序列。
                      </p>
                    </div>
                  )}
                  
                  {selectedPattern.sequence.length >= 4 && (
                    <div className="p-3 bg-blue-50 border border-blue-200 rounded-md">
                      <p className="text-sm text-blue-800">
                        <span className="font-medium">复杂行为链：</span>
                        包含 {selectedPattern.sequence.length} 个步骤的复杂行为序列，建议深入分析用户意图。
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <span className="text-4xl mb-4 block">🔍</span>
              <p>选择左侧的行为模式查看详细分析</p>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
};