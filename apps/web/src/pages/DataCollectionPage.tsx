import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState, useEffect } from 'react';
import {
import { logger } from '../utils/logger'
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Chip,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Checkbox,
  TextField,
  IconButton,
  Tooltip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon
} from '@mui/material';
import {
  PlayArrow as StartIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  HourglassEmpty as PendingIcon,
  Speed as SpeedIcon,
  DataUsage as DataIcon,
  Timeline as TimelineIcon
} from '@mui/icons-material';

interface CollectionJob {
  job_id: string;
  source_id: string;
  source_name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'paused';
  progress: number;
  total_records: number;
  collected_records: number;
  processed_records: number;
  failed_records: number;
  start_time: string;
  estimated_completion?: string;
  error_message?: string;
  processing_rules: string[];
  batch_size: number;
}

interface CollectionStats {
  today_collected: number;
  week_collected: number;
  total_collected: number;
  failed_jobs: number;
  avg_speed: number;
  active_jobs: number;
}

export default function DataCollectionPage() {
  const [jobs, setJobs] = useState<CollectionJob[]>([]);
  const [stats, setStats] = useState<CollectionStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [startJobDialog, setStartJobDialog] = useState(false);
  const [availableSources, setAvailableSources] = useState<any[]>([]);
  const [newJob, setNewJob] = useState({
    source_id: '',
    processing_rules: ['text_cleaning', 'format_standardization'],
    batch_size: 100,
    auto_start: true
  });

  const fetchJobs = async () => {
    try {
      setLoading(true);
      const response = await apiFetch(buildApiUrl('/api/v1/training-data/collection/jobs'));
      const data = await response.json();
      setJobs(data);
    } catch (err) {
      setError('获取收集任务失败');
    } finally {
      setLoading(false);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await apiFetch(buildApiUrl('/api/v1/training-data/collection/stats'));
      const data = await response.json();
      setStats(data);
    } catch (err) {
      logger.error('获取统计信息失败:', err);
    }
  };

  const fetchSources = async () => {
    try {
      const response = await apiFetch(buildApiUrl('/api/v1/training-data/sources?active_only=true'));
      const data = await response.json();
      setAvailableSources(data);
    } catch (err) {
      logger.error('获取数据源失败:', err);
    }
  };

  const handleStartJob = async () => {
    try {
      setLoading(true);
      const response = await apiFetch(buildApiUrl('/api/v1/training-data/collect'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newJob)
      });
      
      await response.json().catch(() => null);
      setStartJobDialog(false);
      setNewJob({
        source_id: '',
        processing_rules: ['text_cleaning', 'format_standardization'],
        batch_size: 100,
        auto_start: true
      });
      await fetchJobs();
    } catch (err) {
      setError(err instanceof Error ? err.message : '启动收集任务失败');
    } finally {
      setLoading(false);
    }
  };

  const handleJobAction = async (jobId: string, action: 'pause' | 'resume' | 'stop') => {
    try {
      const response = await apiFetch(buildApiUrl(`/api/v1/training-data/collection/jobs/${jobId}/${action}`), {
        method: 'POST'
      });
      
      await response.json().catch(() => null);
      await fetchJobs();
    } catch (err) {
      setError(err instanceof Error ? err.message : '操作失败');
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircleIcon color="success" />;
      case 'failed': return <ErrorIcon color="error" />;
      case 'running': return <SpeedIcon color="primary" />;
      case 'paused': return <PauseIcon color="warning" />;
      default: return <PendingIcon color="action" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'failed': return 'error';
      case 'running': return 'primary';
      case 'paused': return 'warning';
      default: return 'default';
    }
  };

  const processingRuleOptions = [
    { value: 'text_cleaning', label: '文本清理' },
    { value: 'deduplication', label: '去重处理' },
    { value: 'format_standardization', label: '格式标准化' },
    { value: 'quality_filtering', label: '质量过滤' },
    { value: 'data_enrichment', label: '数据丰富化' }
  ];

  useEffect(() => {
    fetchJobs();
    fetchStats();
    fetchSources();
    
    const interval = setInterval(() => {
      fetchJobs();
      fetchStats();
    }, 5000); // 每5秒刷新一次

    return () => clearInterval(interval);
  }, []);

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        数据收集
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        从配置的数据源中自动收集数据，支持批量处理、实时监控和错误恢复
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* 统计概览 */}
      {stats && (
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={2}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <DataIcon color="primary" sx={{ mr: 2 }} />
                  <Box>
                    <Typography variant="h6" color="primary">
                      {stats.today_collected}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      今日收集
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={2}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <TimelineIcon color="success" sx={{ mr: 2 }} />
                  <Box>
                    <Typography variant="h6" color="success.main">
                      {stats.week_collected}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      本周收集
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={2}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <SpeedIcon color="info" sx={{ mr: 2 }} />
                  <Box>
                    <Typography variant="h6" color="info.main">
                      {stats.avg_speed}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      条/分钟
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={2}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <PendingIcon color="warning" sx={{ mr: 2 }} />
                  <Box>
                    <Typography variant="h6" color="warning.main">
                      {stats.active_jobs}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      运行中任务
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={2}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <ErrorIcon color="error" sx={{ mr: 2 }} />
                  <Box>
                    <Typography variant="h6" color="error.main">
                      {stats.failed_jobs}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      失败任务
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={2}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <DataIcon color="secondary" sx={{ mr: 2 }} />
                  <Box>
                    <Typography variant="h6" color="secondary.main">
                      {stats.total_collected}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      历史总计
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* 操作工具栏 */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Box>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={() => {
              fetchJobs();
              fetchStats();
            }}
            sx={{ mr: 2 }}
          >
            刷新
          </Button>
        </Box>
        <Button
          variant="contained"
          startIcon={<StartIcon />}
          onClick={() => setStartJobDialog(true)}
          disabled={loading || availableSources.length === 0}
        >
          启动收集任务
        </Button>
      </Box>

      {/* 收集任务列表 */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            收集任务列表
          </Typography>
          <TableContainer component={Paper} variant="outlined">
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>任务信息</TableCell>
                  <TableCell>数据源</TableCell>
                  <TableCell>状态</TableCell>
                  <TableCell>进度</TableCell>
                  <TableCell>收集统计</TableCell>
                  <TableCell>开始时间</TableCell>
                  <TableCell>预计完成</TableCell>
                  <TableCell>操作</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {jobs.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={8} align="center">
                      <Typography variant="body2" color="text.secondary">
                        暂无收集任务，点击"启动收集任务"开始数据收集
                      </Typography>
                    </TableCell>
                  </TableRow>
                ) : (
                  jobs.map((job) => (
                    <TableRow key={job.job_id} hover>
                      <TableCell>
                        <Box>
                          <Typography variant="subtitle2" fontWeight="bold">
                            任务 #{job.job_id.substring(0, 8)}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            批次大小: {job.batch_size}
                          </Typography>
                          {job.processing_rules.length > 0 && (
                            <Box sx={{ mt: 1 }}>
                              {job.processing_rules.slice(0, 2).map(rule => (
                                <Chip
                                  key={rule}
                                  label={rule}
                                  size="small"
                                  variant="outlined"
                                  sx={{ mr: 0.5, fontSize: '0.7rem' }}
                                />
                              ))}
                              {job.processing_rules.length > 2 && (
                                <Chip
                                  label={`+${job.processing_rules.length - 2}`}
                                  size="small"
                                  variant="outlined"
                                  sx={{ fontSize: '0.7rem' }}
                                />
                              )}
                            </Box>
                          )}
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {job.source_name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {job.source_id}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {getStatusIcon(job.status)}
                          <Chip
                            label={job.status}
                            color={getStatusColor(job.status) as any}
                            size="small"
                          />
                        </Box>
                        {job.error_message && (
                          <Tooltip title={job.error_message}>
                            <Typography variant="caption" color="error.main" sx={{ display: 'block' }}>
                              {job.error_message.substring(0, 30)}...
                            </Typography>
                          </Tooltip>
                        )}
                      </TableCell>
                      <TableCell>
                        <Box sx={{ width: 100, mb: 1 }}>
                          <LinearProgress
                            variant="determinate"
                            value={job.progress}
                            color={job.status === 'failed' ? 'error' : 'primary'}
                          />
                        </Box>
                        <Typography variant="caption">
                          {job.progress.toFixed(1)}%
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          收集: {job.collected_records}/{job.total_records}
                        </Typography>
                        <Typography variant="body2" color="success.main">
                          处理: {job.processed_records}
                        </Typography>
                        {job.failed_records > 0 && (
                          <Typography variant="body2" color="error.main">
                            失败: {job.failed_records}
                          </Typography>
                        )}
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {new Date(job.start_time).toLocaleString()}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {job.estimated_completion 
                            ? new Date(job.estimated_completion).toLocaleString()
                            : '-'
                          }
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          {job.status === 'running' && (
                            <Tooltip title="暂停">
                              <IconButton
                                size="small"
                                color="warning"
                                onClick={() => handleJobAction(job.job_id, 'pause')}
                              >
                                <PauseIcon />
                              </IconButton>
                            </Tooltip>
                          )}
                          {job.status === 'paused' && (
                            <Tooltip title="继续">
                              <IconButton
                                size="small"
                                color="primary"
                                onClick={() => handleJobAction(job.job_id, 'resume')}
                              >
                                <StartIcon />
                              </IconButton>
                            </Tooltip>
                          )}
                          {(job.status === 'running' || job.status === 'paused') && (
                            <Tooltip title="停止">
                              <IconButton
                                size="small"
                                color="error"
                                onClick={() => handleJobAction(job.job_id, 'stop')}
                              >
                                <StopIcon />
                              </IconButton>
                            </Tooltip>
                          )}
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* 启动收集任务对话框 */}
      <Dialog open={startJobDialog} onClose={() => setStartJobDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>启动数据收集任务</DialogTitle>
        <DialogContent>
          <FormControl fullWidth margin="normal">
            <InputLabel>选择数据源</InputLabel>
            <Select
              value={newJob.source_id}
              label="选择数据源"
              onChange={(e) => setNewJob(prev => ({ ...prev, source_id: e.target.value }))}
            >
              {availableSources.map((source) => (
                <MenuItem key={source.source_id} value={source.source_id}>
                  {source.name} ({source.source_type})
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
            预处理规则
          </Typography>
          <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
            {processingRuleOptions.map((rule) => (
              <FormControlLabel
                key={rule.value}
                control={
                  <Checkbox
                    checked={newJob.processing_rules.includes(rule.value)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setNewJob(prev => ({
                          ...prev,
                          processing_rules: [...prev.processing_rules, rule.value]
                        }));
                      } else {
                        setNewJob(prev => ({
                          ...prev,
                          processing_rules: prev.processing_rules.filter(r => r !== rule.value)
                        }));
                      }
                    }}
                  />
                }
                label={rule.label}
              />
            ))}
          </Paper>

          <TextField
            fullWidth
            margin="normal"
            label="批次大小"
            type="number"
            value={newJob.batch_size}
            onChange={(e) => setNewJob(prev => ({ ...prev, batch_size: parseInt(e.target.value) || 100 }))}
            helperText="每批次处理的记录数量，建议100-1000"
            inputProps={{ min: 10, max: 10000 }}
          />

          <FormControlLabel
            control={
              <Checkbox
                checked={newJob.auto_start}
                onChange={(e) => setNewJob(prev => ({ ...prev, auto_start: e.target.checked }))}
              />
            }
            label="自动开始处理"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setStartJobDialog(false)}>取消</Button>
          <Button 
            onClick={handleStartJob} 
            variant="contained"
            disabled={!newJob.source_id || loading}
          >
            启动任务
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
