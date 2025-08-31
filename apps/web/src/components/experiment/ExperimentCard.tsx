/**
 * 实验卡片组件
 */
import React from 'react';
import {
  Card,
  CardContent,
  CardActions,
  Typography,
  Chip,
  Box,
  Button,
  LinearProgress,
  Stack,
  IconButton,
  Tooltip,
  Avatar,
  AvatarGroup,
  // Grid  // 暂时注释未使用的导入
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  ContentCopy as CopyIcon,
  Timeline as TimelineIcon,
  Group as GroupIcon,
  Science as ScienceIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon
} from '@mui/icons-material';
import { /* format, */ formatDistanceToNow } from 'date-fns'; // 暂时注释未使用的format
import { zhCN } from 'date-fns/locale';

export interface ExperimentData {
  id: string;
  name: string;
  description: string;
  type: string;
  status: 'draft' | 'active' | 'paused' | 'completed' | 'archived';
  startDate?: Date;
  endDate?: Date;
  variants: {
    id: string;
    name: string;
    traffic: number;
    isControl: boolean;
  }[];
  metrics: {
    name: string;
    type: 'primary' | 'secondary' | 'guardrail';
    currentValue?: number;
    baselineValue?: number;
    improvement?: number;
    pValue?: number;
    significant?: boolean;
  }[];
  sampleSize?: {
    current: number;
    required: number;
  };
  owners?: string[];
  tags?: string[];
  healthStatus?: 'healthy' | 'warning' | 'error';
  healthMessage?: string;
}

interface ExperimentCardProps {
  experiment: ExperimentData;
  onAction?: (action: string, experimentId: string) => void;
  showDetails?: boolean;
  compact?: boolean;
}

const ExperimentCard: React.FC<ExperimentCardProps> = ({
  experiment,
  onAction,
  showDetails = true,
  compact = false
}) => {
  // 获取状态颜色
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'success';
      case 'paused':
        return 'warning';
      case 'completed':
        return 'info';
      case 'archived':
        return 'default';
      default:
        return 'default';
    }
  };

  // 获取状态标签
  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'draft':
        return '草稿';
      case 'active':
        return '运行中';
      case 'paused':
        return '已暂停';
      case 'completed':
        return '已完成';
      case 'archived':
        return '已归档';
      default:
        return status;
    }
  };

  // 获取健康状态图标
  const getHealthIcon = () => {
    switch (experiment.healthStatus) {
      case 'healthy':
        return <CheckCircleIcon color="success" fontSize="small" />;
      case 'warning':
        return <WarningIcon color="warning" fontSize="small" />;
      case 'error':
        return <ErrorIcon color="error" fontSize="small" />;
      default:
        return null;
    }
  };

  // 计算实验进度
  const getProgress = () => {
    if (experiment.sampleSize) {
      return Math.min(
        (experiment.sampleSize.current / experiment.sampleSize.required) * 100,
        100
      );
    }
    if (experiment.startDate && experiment.endDate) {
      const total = experiment.endDate.getTime() - experiment.startDate.getTime();
      const current = Date.now() - experiment.startDate.getTime();
      return Math.min((current / total) * 100, 100);
    }
    return 0;
  };

  // 获取主要指标
  const primaryMetric = experiment.metrics.find(m => m.type === 'primary');

  return (
    <Card
      sx={{
        height: compact ? 'auto' : '100%',
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
        '&:hover': {
          boxShadow: 3
        }
      }}
    >
      {/* 状态标签 */}
      <Box sx={{ position: 'absolute', top: 16, right: 16, zIndex: 1 }}>
        <Chip
          label={getStatusLabel(experiment.status)}
          color={getStatusColor(experiment.status)}
          size="small"
        />
      </Box>

      <CardContent sx={{ flexGrow: 1 }}>
        {/* 标题行 */}
        <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 2 }}>
          <Avatar sx={{ bgcolor: 'primary.main', mr: 2 }}>
            <ScienceIcon />
          </Avatar>
          <Box sx={{ flexGrow: 1 }}>
            <Typography variant="h6" component="div" gutterBottom>
              {experiment.name}
              {getHealthIcon() && (
                <Tooltip title={experiment.healthMessage || ''}>
                  <Box component="span" sx={{ ml: 1 }}>
                    {getHealthIcon()}
                  </Box>
                </Tooltip>
              )}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {experiment.description}
            </Typography>
          </Box>
        </Box>

        {/* 实验类型和时间 */}
        <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
          <Chip
            label={experiment.type}
            size="small"
            variant="outlined"
            icon={<ScienceIcon />}
          />
          {experiment.startDate && (
            <Typography variant="caption" color="text.secondary">
              开始于 {formatDistanceToNow(experiment.startDate, { locale: zhCN, addSuffix: true })}
            </Typography>
          )}
        </Stack>

        {showDetails && (
          <>
            {/* 变体信息 */}
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                变体配置
              </Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap">
                {experiment.variants.map(variant => (
                  <Chip
                    key={variant.id}
                    label={`${variant.name}: ${variant.traffic}%`}
                    size="small"
                    color={variant.isControl ? 'primary' : 'default'}
                    variant={variant.isControl ? 'filled' : 'outlined'}
                  />
                ))}
              </Stack>
            </Box>

            {/* 主要指标 */}
            {primaryMetric && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  主要指标: {primaryMetric.name}
                </Typography>
                {primaryMetric.improvement !== undefined && (
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography
                      variant="h6"
                      color={primaryMetric.improvement > 0 ? 'success.main' : 'error.main'}
                    >
                      {primaryMetric.improvement > 0 ? '+' : ''}{primaryMetric.improvement.toFixed(2)}%
                    </Typography>
                    {primaryMetric.pValue !== undefined && (
                      <Chip
                        label={`p=${primaryMetric.pValue.toFixed(3)}`}
                        size="small"
                        color={primaryMetric.significant ? 'success' : 'default'}
                        variant={primaryMetric.significant ? 'filled' : 'outlined'}
                      />
                    )}
                  </Box>
                )}
              </Box>
            )}

            {/* 进度条 */}
            {experiment.status === 'active' && (
              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    实验进度
                  </Typography>
                  {experiment.sampleSize && (
                    <Typography variant="body2" color="text.secondary">
                      {experiment.sampleSize.current.toLocaleString()} / {experiment.sampleSize.required.toLocaleString()}
                    </Typography>
                  )}
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={getProgress()}
                  sx={{ height: 6, borderRadius: 3 }}
                />
              </Box>
            )}

            {/* 标签 */}
            {experiment.tags && experiment.tags.length > 0 && (
              <Stack direction="row" spacing={0.5} flexWrap="wrap" sx={{ mb: 1 }}>
                {experiment.tags.map(tag => (
                  <Chip
                    key={tag}
                    label={tag}
                    size="small"
                    variant="outlined"
                    sx={{ mb: 0.5 }}
                  />
                ))}
              </Stack>
            )}

            {/* 负责人 */}
            {experiment.owners && experiment.owners.length > 0 && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <GroupIcon fontSize="small" color="action" />
                <AvatarGroup max={3} sx={{ '& .MuiAvatar-root': { width: 24, height: 24, fontSize: 12 } }}>
                  {experiment.owners.map((owner, index) => (
                    <Avatar key={index} sx={{ bgcolor: 'primary.main' }}>
                      {owner[0].toUpperCase()}
                    </Avatar>
                  ))}
                </AvatarGroup>
              </Box>
            )}
          </>
        )}
      </CardContent>

      {/* 操作按钮 */}
      {onAction && (
        <CardActions sx={{ justifyContent: 'space-between', px: 2, pb: 2 }}>
          <Stack direction="row" spacing={1}>
            {experiment.status === 'draft' && (
              <Button
                size="small"
                startIcon={<PlayIcon />}
                onClick={() => onAction('start', experiment.id)}
                variant="contained"
                color="primary"
              >
                启动
              </Button>
            )}
            {experiment.status === 'active' && (
              <Button
                size="small"
                startIcon={<PauseIcon />}
                onClick={() => onAction('pause', experiment.id)}
                variant="outlined"
              >
                暂停
              </Button>
            )}
            {experiment.status === 'paused' && (
              <Button
                size="small"
                startIcon={<PlayIcon />}
                onClick={() => onAction('resume', experiment.id)}
                variant="contained"
                color="primary"
              >
                恢复
              </Button>
            )}
            {(experiment.status === 'active' || experiment.status === 'paused') && (
              <Button
                size="small"
                startIcon={<StopIcon />}
                onClick={() => onAction('stop', experiment.id)}
                color="error"
              >
                停止
              </Button>
            )}
          </Stack>
          <Stack direction="row" spacing={0}>
            <Tooltip title="查看报告">
              <IconButton
                size="small"
                onClick={() => onAction('report', experiment.id)}
              >
                <TimelineIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="编辑">
              <IconButton
                size="small"
                onClick={() => onAction('edit', experiment.id)}
                disabled={experiment.status === 'active'}
              >
                <EditIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="复制">
              <IconButton
                size="small"
                onClick={() => onAction('clone', experiment.id)}
              >
                <CopyIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="删除">
              <IconButton
                size="small"
                onClick={() => onAction('delete', experiment.id)}
                disabled={experiment.status === 'active'}
              >
                <DeleteIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Stack>
        </CardActions>
      )}
    </Card>
  );
};

export default ExperimentCard;