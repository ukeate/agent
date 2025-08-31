import React, { useState, useEffect } from 'react';
import {
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
  TextField,
  Avatar,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  IconButton,
  Tooltip,
  Tabs,
  Tab,
  Badge,
  Divider
} from '@mui/material';
import {
  Add as AddIcon,
  Refresh as RefreshIcon,
  Edit as EditIcon,
  Assignment as AssignmentIcon,
  Person as PersonIcon,
  Speed as SpeedIcon,
  CheckCircle as CheckCircleIcon,
  Schedule as ScheduleIcon,
  Star as StarIcon,
  TrendingUp as TrendingUpIcon,
  Group as GroupIcon,
  Assessment as AssessmentIcon,
  Visibility as VisibilityIcon,
  PlayArrow as StartIcon,
  Pause as PauseIcon
} from '@mui/icons-material';

interface AnnotationTask {
  id: string;
  task_id: string;
  name: string;
  description: string;
  task_type: string;
  data_record_count: number;
  assignee_count: number;
  completed_annotations: number;
  status: string;
  created_by: string;
  created_at: string;
  deadline?: string;
  progress: number;
  quality_score: number;
}

interface Annotator {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  skill_level: number;
  specialties: string[];
  stats: {
    total_annotations: number;
    avg_time_per_annotation: number;
    accuracy_rate: number;
    consistency_score: number;
  };
  status: 'online' | 'offline' | 'busy';
  last_active: string;
}

interface AnnotationStats {
  total_tasks: number;
  active_tasks: number;
  completed_tasks: number;
  total_annotators: number;
  active_annotators: number;
  avg_completion_time: number;
  overall_quality: number;
  productivity_trend: number;
}

export default function DataAnnotationManagementPage() {
  const [currentTab, setCurrentTab] = useState(0);
  const [tasks, setTasks] = useState<AnnotationTask[]>([]);
  const [annotators, setAnnotators] = useState<Annotator[]>([]);
  const [stats, setStats] = useState<AnnotationStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [createTaskDialog, setCreateTaskDialog] = useState(false);
  const [assignDialog, setAssignDialog] = useState(false);
  const [selectedTask, setSelectedTask] = useState<AnnotationTask | null>(null);
  const [newTask, setNewTask] = useState({
    name: '',
    description: '',
    task_type: 'text_classification',
    data_records: [] as string[],
    annotation_schema: {},
    guidelines: '',
    assignees: [] as string[]
  });

  const fetchTasks = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/training-data/annotation-tasks');
      if (response.ok) {
        const data = await response.json();
        setTasks(data);
      }
    } catch (err) {
      setError('获取标注任务失败');
    } finally {
      setLoading(false);
    }
  };

  const fetchAnnotators = async () => {
    try {
      const response = await fetch('/api/v1/training-data/annotators');
      if (response.ok) {
        const data = await response.json();
        setAnnotators(data);
      }
    } catch (err) {
      console.error('获取标注者信息失败:', err);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch('/api/v1/training-data/annotation/stats');
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (err) {
      console.error('获取统计信息失败:', err);
    }
  };

  const handleCreateTask = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/training-data/annotation-tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newTask)
      });
      
      if (response.ok) {
        setCreateTaskDialog(false);
        setNewTask({
          name: '',
          description: '',
          task_type: 'text_classification',
          data_records: [],
          annotation_schema: {},
          guidelines: '',
          assignees: []
        });
        await fetchTasks();
      } else {
        throw new Error('创建标注任务失败');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '创建标注任务失败');
    } finally {
      setLoading(false);
    }
  };

  const handleAssignTask = async (taskId: string, annotatorIds: string[]) => {
    try {
      const response = await fetch(`/api/v1/training-data/annotation-tasks/${taskId}/assign`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ annotator_ids: annotatorIds })
      });
      
      if (response.ok) {
        await fetchTasks();
        setAssignDialog(false);
      } else {
        throw new Error('分配任务失败');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '分配任务失败');
    }
  };

  const getTaskTypeLabel = (type: string) => {
    const types: Record<string, string> = {
      'text_classification': '文本分类',
      'sequence_labeling': '序列标注',
      'question_answering': '问答标注',
      'sentiment_analysis': '情感分析',
      'entity_recognition': '实体识别'
    };
    return types[type] || type;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'in_progress': return 'primary';
      case 'pending': return 'warning';
      case 'reviewed': return 'info';
      default: return 'default';
    }
  };

  const getSkillLevelColor = (level: number) => {
    if (level >= 4.5) return 'success';
    if (level >= 3.5) return 'warning';
    return 'default';
  };

  // 模拟数据
  useEffect(() => {
    const mockTasks: AnnotationTask[] = [
      {
        id: '1',
        task_id: 'task_001',
        name: '新闻文本分类',
        description: '对新闻文章进行类别分类',
        task_type: 'text_classification',
        data_record_count: 1000,
        assignee_count: 3,
        completed_annotations: 750,
        status: 'in_progress',
        created_by: 'admin',
        created_at: '2024-01-20T10:00:00Z',
        deadline: '2024-01-25T18:00:00Z',
        progress: 75,
        quality_score: 0.92
      },
      {
        id: '2',
        task_id: 'task_002',
        name: '产品评论情感分析',
        description: '分析用户评论的情感倾向',
        task_type: 'sentiment_analysis',
        data_record_count: 500,
        assignee_count: 2,
        completed_annotations: 150,
        status: 'in_progress',
        created_by: 'admin',
        created_at: '2024-01-18T14:30:00Z',
        progress: 30,
        quality_score: 0.88
      }
    ];

    const mockAnnotators: Annotator[] = [
      {
        id: '1',
        name: '张小明',
        email: 'zhang@example.com',
        skill_level: 4.5,
        specialties: ['文本分类', '情感分析'],
        stats: {
          total_annotations: 1250,
          avg_time_per_annotation: 45,
          accuracy_rate: 0.94,
          consistency_score: 0.91
        },
        status: 'online',
        last_active: '2024-01-20T15:30:00Z'
      },
      {
        id: '2',
        name: '李小红',
        email: 'li@example.com',
        skill_level: 4.2,
        specialties: ['序列标注', '实体识别'],
        stats: {
          total_annotations: 980,
          avg_time_per_annotation: 52,
          accuracy_rate: 0.91,
          consistency_score: 0.89
        },
        status: 'online',
        last_active: '2024-01-20T15:25:00Z'
      },
      {
        id: '3',
        name: '王大伟',
        email: 'wang@example.com',
        skill_level: 3.8,
        specialties: ['问答标注'],
        stats: {
          total_annotations: 650,
          avg_time_per_annotation: 65,
          accuracy_rate: 0.87,
          consistency_score: 0.85
        },
        status: 'busy',
        last_active: '2024-01-20T14:45:00Z'
      }
    ];

    const mockStats: AnnotationStats = {
      total_tasks: 12,
      active_tasks: 5,
      completed_tasks: 7,
      total_annotators: 8,
      active_annotators: 5,
      avg_completion_time: 3.2,
      overall_quality: 0.91,
      productivity_trend: 0.15
    };

    setTasks(mockTasks);
    setAnnotators(mockAnnotators);
    setStats(mockStats);
    
    fetchTasks();
    fetchAnnotators();
    fetchStats();
  }, []);

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        数据标注管理
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        管理标注任务、分配标注者、监控标注进度和质量
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* 统计概览 */}
      {stats && (
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <AssignmentIcon color="primary" sx={{ mr: 2 }} />
                  <Box>
                    <Typography variant="h6" color="primary">
                      {stats.total_tasks}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      总任务数
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <SpeedIcon color="success" sx={{ mr: 2 }} />
                  <Box>
                    <Typography variant="h6" color="success.main">
                      {stats.active_tasks}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      进行中任务
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <GroupIcon color="info" sx={{ mr: 2 }} />
                  <Box>
                    <Typography variant="h6" color="info.main">
                      {stats.active_annotators}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      在线标注者
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <StarIcon color="warning" sx={{ mr: 2 }} />
                  <Box>
                    <Typography variant="h6" color="warning.main">
                      {(stats.overall_quality * 100).toFixed(1)}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      整体质量
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* 标签页 */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={currentTab} onChange={(e, newValue) => setCurrentTab(newValue)}>
          <Tab label="标注任务" />
          <Tab label="标注者管理" />
          <Tab label="进度监控" />
        </Tabs>
      </Box>

      {/* 标注任务列表 */}
      {currentTab === 0 && (
        <Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={fetchTasks}
            >
              刷新
            </Button>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={() => setCreateTaskDialog(true)}
            >
              创建标注任务
            </Button>
          </Box>

          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                标注任务列表
              </Typography>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>任务信息</TableCell>
                      <TableCell>类型</TableCell>
                      <TableCell>进度</TableCell>
                      <TableCell>质量</TableCell>
                      <TableCell>分配情况</TableCell>
                      <TableCell>截止时间</TableCell>
                      <TableCell>操作</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {tasks.map((task) => (
                      <TableRow key={task.id} hover>
                        <TableCell>
                          <Box>
                            <Typography variant="subtitle2" fontWeight="bold">
                              {task.name}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {task.description}
                            </Typography>
                            <br />
                            <Typography variant="caption" color="text.secondary">
                              记录数: {task.data_record_count}
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={getTaskTypeLabel(task.task_type)}
                            size="small"
                            variant="outlined"
                          />
                        </TableCell>
                        <TableCell>
                          <Box sx={{ minWidth: 120 }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                              <LinearProgress
                                variant="determinate"
                                value={task.progress}
                                sx={{ width: 80, mr: 1 }}
                              />
                              <Typography variant="body2">
                                {task.progress}%
                              </Typography>
                            </Box>
                            <Typography variant="caption" color="text.secondary">
                              {task.completed_annotations}/{task.data_record_count}
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <StarIcon 
                              color={task.quality_score >= 0.9 ? 'success' : 'warning'}
                              sx={{ mr: 1 }}
                            />
                            <Typography variant="body2">
                              {(task.quality_score * 100).toFixed(1)}%
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <GroupIcon sx={{ mr: 1, color: 'text.secondary' }} />
                            <Typography variant="body2">
                              {task.assignee_count} 人
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          {task.deadline ? (
                            <Typography variant="body2">
                              {new Date(task.deadline).toLocaleDateString()}
                            </Typography>
                          ) : (
                            <Typography variant="body2" color="text.secondary">
                              无期限
                            </Typography>
                          )}
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', gap: 1 }}>
                            <Tooltip title="查看详情">
                              <IconButton size="small" color="primary">
                                <VisibilityIcon />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="分配标注者">
                              <IconButton 
                                size="small" 
                                color="default"
                                onClick={() => {
                                  setSelectedTask(task);
                                  setAssignDialog(true);
                                }}
                              >
                                <AssignmentIcon />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="编辑任务">
                              <IconButton size="small" color="default">
                                <EditIcon />
                              </IconButton>
                            </Tooltip>
                          </Box>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Box>
      )}

      {/* 标注者管理 */}
      {currentTab === 1 && (
        <Box>
          <Typography variant="h6" gutterBottom>
            标注者列表
          </Typography>
          <Grid container spacing={3}>
            {annotators.map((annotator) => (
              <Grid item xs={12} md={6} lg={4} key={annotator.id}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Badge
                        color={annotator.status === 'online' ? 'success' : 'default'}
                        badgeContent=" "
                        variant="dot"
                      >
                        <Avatar sx={{ mr: 2 }}>
                          <PersonIcon />
                        </Avatar>
                      </Badge>
                      <Box>
                        <Typography variant="subtitle1" fontWeight="bold">
                          {annotator.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {annotator.email}
                        </Typography>
                      </Box>
                    </Box>

                    <Box sx={{ mb: 2 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <Typography variant="body2" sx={{ mr: 1 }}>
                          技能水平:
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          {[1, 2, 3, 4, 5].map((star) => (
                            <StarIcon
                              key={star}
                              sx={{
                                fontSize: 16,
                                color: star <= annotator.skill_level ? 'gold' : 'lightgray'
                              }}
                            />
                          ))}
                          <Typography variant="body2" sx={{ ml: 1 }}>
                            {annotator.skill_level.toFixed(1)}
                          </Typography>
                        </Box>
                      </Box>
                      
                      <Box sx={{ mb: 1 }}>
                        <Typography variant="body2" gutterBottom>
                          专业领域:
                        </Typography>
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {annotator.specialties.map((specialty) => (
                            <Chip
                              key={specialty}
                              label={specialty}
                              size="small"
                              variant="outlined"
                            />
                          ))}
                        </Box>
                      </Box>
                    </Box>

                    <Divider sx={{ my: 2 }} />

                    <Box>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        绩效统计
                      </Typography>
                      <Grid container spacing={1}>
                        <Grid item xs={6}>
                          <Typography variant="caption">
                            总标注: {annotator.stats.total_annotations}
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="caption">
                            平均用时: {annotator.stats.avg_time_per_annotation}s
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="caption">
                            准确率: {(annotator.stats.accuracy_rate * 100).toFixed(1)}%
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="caption">
                            一致性: {(annotator.stats.consistency_score * 100).toFixed(1)}%
                          </Typography>
                        </Grid>
                      </Grid>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>
      )}

      {/* 进度监控 */}
      {currentTab === 2 && (
        <Box>
          <Typography variant="h6" gutterBottom>
            整体进度监控
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    任务完成情况
                  </Typography>
                  {tasks.map((task) => (
                    <Box key={task.id} sx={{ mb: 2 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2">
                          {task.name}
                        </Typography>
                        <Typography variant="body2">
                          {task.progress}%
                        </Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={task.progress}
                        color={task.progress >= 90 ? 'success' : task.progress >= 50 ? 'primary' : 'warning'}
                      />
                    </Box>
                  ))}
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    标注者活跃度
                  </Typography>
                  <List>
                    {annotators.map((annotator) => (
                      <ListItem key={annotator.id}>
                        <ListItemAvatar>
                          <Badge
                            color={annotator.status === 'online' ? 'success' : 'default'}
                            badgeContent=" "
                            variant="dot"
                          >
                            <Avatar>
                              <PersonIcon />
                            </Avatar>
                          </Badge>
                        </ListItemAvatar>
                        <ListItemText
                          primary={annotator.name}
                          secondary={
                            <Box>
                              <Typography variant="caption">
                                状态: {annotator.status} | 
                                最后活跃: {new Date(annotator.last_active).toLocaleString()}
                              </Typography>
                            </Box>
                          }
                        />
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Box>
      )}

      {/* 创建标注任务对话框 */}
      <Dialog open={createTaskDialog} onClose={() => setCreateTaskDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>创建标注任务</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            margin="normal"
            label="任务名称"
            value={newTask.name}
            onChange={(e) => setNewTask(prev => ({ ...prev, name: e.target.value }))}
          />
          <TextField
            fullWidth
            margin="normal"
            label="任务描述"
            multiline
            rows={3}
            value={newTask.description}
            onChange={(e) => setNewTask(prev => ({ ...prev, description: e.target.value }))}
          />
          <FormControl fullWidth margin="normal">
            <InputLabel>任务类型</InputLabel>
            <Select
              value={newTask.task_type}
              label="任务类型"
              onChange={(e) => setNewTask(prev => ({ ...prev, task_type: e.target.value }))}
            >
              <MenuItem value="text_classification">文本分类</MenuItem>
              <MenuItem value="sequence_labeling">序列标注</MenuItem>
              <MenuItem value="question_answering">问答标注</MenuItem>
              <MenuItem value="sentiment_analysis">情感分析</MenuItem>
              <MenuItem value="entity_recognition">实体识别</MenuItem>
            </Select>
          </FormControl>
          <TextField
            fullWidth
            margin="normal"
            label="标注指南"
            multiline
            rows={4}
            value={newTask.guidelines}
            onChange={(e) => setNewTask(prev => ({ ...prev, guidelines: e.target.value }))}
            helperText="详细的标注说明和示例"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateTaskDialog(false)}>取消</Button>
          <Button 
            onClick={handleCreateTask} 
            variant="contained"
            disabled={!newTask.name || loading}
          >
            创建任务
          </Button>
        </DialogActions>
      </Dialog>

      {/* 分配标注者对话框 */}
      <Dialog open={assignDialog} onClose={() => setAssignDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>分配标注者</DialogTitle>
        <DialogContent>
          {selectedTask && (
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                任务: {selectedTask.name}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                类型: {getTaskTypeLabel(selectedTask.task_type)} | 
                记录数: {selectedTask.data_record_count}
              </Typography>
              
              <Typography variant="subtitle2" gutterBottom>
                选择标注者:
              </Typography>
              <List>
                {annotators.map((annotator) => (
                  <ListItem key={annotator.id}>
                    <ListItemAvatar>
                      <Avatar>
                        <PersonIcon />
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary={annotator.name}
                      secondary={
                        <Box>
                          <Typography variant="caption">
                            技能: {annotator.skill_level}/5 | 
                            专业: {annotator.specialties.join(', ')}
                          </Typography>
                        </Box>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAssignDialog(false)}>取消</Button>
          <Button variant="contained">
            确认分配
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}