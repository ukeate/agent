import React from 'react';
import { Card } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Progress } from '../ui/lprogress';

interface ExperimentCardProps {
  experiment: {
    id: string;
    name: string;
    status: string;
    algorithm: string;
    objective: string;
    created_at: string;
    best_value?: number;
    total_trials?: number;
    successful_trials?: number;
  };
  onView?: (id: string) => void;
  onStart?: (id: string) => void;
  onStop?: (id: string) => void;
  onDelete?: (id: string) => void;
}

const statusColors = {
  'created': 'bg-gray-500',
  'running': 'bg-blue-500',
  'completed': 'bg-green-500',
  'failed': 'bg-red-500',
  'stopped': 'bg-yellow-500'
};

const algorithmLabels = {
  'tpe': 'TPE',
  'cmaes': 'CMA-ES',
  'random': '随机搜索',
  'grid': '网格搜索',
  'nsga2': 'NSGA-II'
};

export const ExperimentCard: React.FC<ExperimentCardProps> = ({
  experiment,
  onView,
  onStart,
  onStop,
  onDelete
}) => {
  const getStatusColor = (status: string) => statusColors[status as keyof typeof statusColors] || 'bg-gray-500';
  const getAlgorithmLabel = (algorithm: string) => algorithmLabels[algorithm as keyof typeof algorithmLabels] || algorithm.toUpperCase();
  
  const successRate = experiment.total_trials && experiment.successful_trials 
    ? (experiment.successful_trials / experiment.total_trials) * 100 
    : 0;

  return (
    <Card className="p-6 hover:shadow-lg transition-shadow">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 mb-1">
            {experiment.name}
          </h3>
          <div className="flex items-center space-x-2">
            <Badge className={`${getStatusColor(experiment.status)} text-white text-xs`}>
              {experiment.status}
            </Badge>
            <Badge className="bg-blue-100 text-blue-800 text-xs">
              {getAlgorithmLabel(experiment.algorithm)}
            </Badge>
            <Badge className="bg-purple-100 text-purple-800 text-xs">
              {experiment.objective === 'maximize' ? '最大化' : '最小化'}
            </Badge>
          </div>
        </div>
        <div className="flex space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => onView?.(experiment.id)}
          >
            查看
          </Button>
          {experiment.status === 'created' && (
            <Button
              size="sm"
              onClick={() => onStart?.(experiment.id)}
            >
              启动
            </Button>
          )}
          {experiment.status === 'running' && (
            <Button
              variant="destructive"
              size="sm"
              onClick={() => onStop?.(experiment.id)}
            >
              停止
            </Button>
          )}
          {['completed', 'failed', 'stopped'].includes(experiment.status) && (
            <Button
              variant="destructive"
              size="sm"
              onClick={() => onDelete?.(experiment.id)}
            >
              删除
            </Button>
          )}
        </div>
      </div>

      <div className="space-y-3">
        {experiment.best_value !== undefined && (
          <div>
            <span className="text-sm text-gray-600">最佳值: </span>
            <span className="font-mono text-sm font-medium">
              {experiment.best_value.toFixed(6)}
            </span>
          </div>
        )}

        {experiment.total_trials !== undefined && (
          <div>
            <div className="flex justify-between text-sm text-gray-600 mb-1">
              <span>试验进度</span>
              <span>{experiment.successful_trials || 0}/{experiment.total_trials}</span>
            </div>
            <Progress value={successRate} className="h-2" />
          </div>
        )}

        <div className="text-xs text-gray-500">
          创建时间: {new Date(experiment.created_at).toLocaleString('zh-CN')}
        </div>
      </div>
    </Card>
  );
};