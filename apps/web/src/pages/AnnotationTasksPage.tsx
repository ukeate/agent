import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  LinearProgress,
  Chip,
  Alert,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';
import {
  Add as AddIcon,
  Refresh as RefreshIcon,
  Visibility as VisibilityIcon,
  Edit as EditIcon,
  PlayArrow as StartIcon,
  Pause as PauseIcon,
  CheckCircle as CheckCircleIcon,
  Assignment as AssignmentIcon,
  Assessment as AssessmentIcon
} from '@mui/icons-material';

export default function AnnotationTasksPage() {
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [createDialog, setCreateDialog] = useState(false);
  const [progressDialog, setProgressDialog] = useState(false);
  const [selectedTask, setSelectedTask] = useState<any>(null);

  const mockTasks = [
    {
      task_id: 'task_001',
      name: '新闻文本分类任务',
      description: '对新闻文章进行主题分类',
      task_type: 'text_classification',
      status: 'in_progress',
      progress: 65,
      total_records: 2000,
      completed_records: 1300,
      assignees: ['张三', '李四', '王五'],
      created_at: '2024-01-15',
      deadline: '2024-01-25',
      quality_score: 0.89
    },
    {
      task_id: 'task_002', 
      name: '电商评论情感分析',
      description: '分析用户评论的情感极性',
      task_type: 'sentiment_analysis',
      status: 'completed',
      progress: 100,
      total_records: 1500,
      completed_records: 1500,
      assignees: ['赵六', '孙七'],
      created_at: '2024-01-10',
      deadline: '2024-01-20',
      quality_score: 0.92
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'in_progress': return 'primary';
      case 'pending': return 'warning';
      default: return 'default';
    }
  };

  const getTaskTypeLabel = (type: string) => {
    const types: Record<string, string> = {
      'text_classification': '文本分类',
      'sequence_labeling': '序列标注',
      'sentiment_analysis': '情感分析',
      'question_answering': '问答标注'
    };
    return types[type] || type;
  };

  useEffect(() => {
    setTasks(mockTasks);
  }, []);

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        标注任务
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        查看和管理所有标注任务的详细信息、进度和质量
      </Typography>

      {/* 统计卡片 */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <AssignmentIcon color="primary" sx={{ mr: 2 }} />
                <Box>
                  <Typography variant="h6" color="primary">
                    {mockTasks.length}
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
                <StartIcon color="warning" sx={{ mr: 2 }} />
                <Box>
                  <Typography variant="h6" color="warning.main">
                    {mockTasks.filter(t => t.status === 'in_progress').length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    进行中
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
                <CheckCircleIcon color="success" sx={{ mr: 2 }} />
                <Box>
                  <Typography variant="h6" color="success.main">
                    {mockTasks.filter(t => t.status === 'completed').length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    已完成
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
                <AssessmentIcon color="info" sx={{ mr: 2 }} />
                <Box>
                  <Typography variant="h6" color="info.main">
                    {(mockTasks.reduce((sum, task) => sum + task.quality_score, 0) / mockTasks.length * 100).toFixed(0)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    平均质量
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* 工具栏 */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={() => {}}
        >
          刷新
        </Button>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setCreateDialog(true)}
        >
          创建新任务
        </Button>
      </Box>

      {/* 任务列表 */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            任务列表
          </Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>任务信息</TableCell>
                  <TableCell>类型</TableCell>
                  <TableCell>状态</TableCell>
                  <TableCell>进度</TableCell>
                  <TableCell>分配人员</TableCell>
                  <TableCell>质量评分</TableCell>
                  <TableCell>截止时间</TableCell>
                  <TableCell>操作</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {tasks.map((task: any) => (
                  <TableRow key={task.task_id} hover>
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
                          ID: {task.task_id} | 创建: {task.created_at}
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
                      <Chip
                        label={task.status}
                        color={getStatusColor(task.status) as any}
                        size="small"
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
                          {task.completed_records}/{task.total_records}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {task.assignees.slice(0, 2).map((assignee: string) => (
                          <Chip
                            key={assignee}
                            label={assignee}
                            size="small"
                            variant="outlined"
                          />
                        ))}
                        {task.assignees.length > 2 && (
                          <Chip
                            label={`+${task.assignees.length - 2}`}
                            size="small"
                            variant="outlined"
                          />
                        )}
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Typography variant="body2" color={task.quality_score >= 0.9 ? 'success.main' : 'text.primary'}>
                          {(task.quality_score * 100).toFixed(0)}%
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {task.deadline}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', gap: 0.5 }}>
                        <Tooltip title="查看详情">
                          <IconButton
                            size="small"
                            color="primary"
                            onClick={() => {
                              setSelectedTask(task);
                              setProgressDialog(true);
                            }}
                          >
                            <VisibilityIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="编辑任务">
                          <IconButton size="small" color="default">
                            <EditIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="质量报告">
                          <IconButton size="small" color="info">
                            <AssessmentIcon />
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

      {/* 创建任务对话框 */}
      <Dialog open={createDialog} onClose={() => setCreateDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>创建标注任务</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            margin="normal"
            label="任务名称"
            placeholder="输入任务名称"
          />
          <TextField
            fullWidth
            margin="normal"
            label="任务描述"
            multiline
            rows={3}
            placeholder="详细描述标注任务的目标和要求"
          />
          <FormControl fullWidth margin="normal">
            <InputLabel>任务类型</InputLabel>
            <Select defaultValue="" label="任务类型">
              <MenuItem value="text_classification">文本分类</MenuItem>
              <MenuItem value="sequence_labeling">序列标注</MenuItem>
              <MenuItem value="sentiment_analysis">情感分析</MenuItem>
              <MenuItem value="question_answering">问答标注</MenuItem>
            </Select>
          </FormControl>
          <TextField
            fullWidth
            margin="normal"
            label="标注指南"
            multiline
            rows={4}
            placeholder="提供详细的标注指南和示例"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialog(false)}>取消</Button>
          <Button variant="contained">创建任务</Button>
        </DialogActions>
      </Dialog>

      {/* 任务详情对话框 */}
      <Dialog open={progressDialog} onClose={() => setProgressDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>任务详情</DialogTitle>
        <DialogContent>
          {selectedTask && (
            <Box>
              <Typography variant="h6" gutterBottom>
                {selectedTask.name}
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                {selectedTask.description}
              </Typography>
              
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">任务类型</Typography>
                  <Typography variant="body2">{getTaskTypeLabel(selectedTask.task_type)}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">当前状态</Typography>
                  <Chip
                    label={selectedTask.status}
                    color={getStatusColor(selectedTask.status) as any}
                    size="small"
                  />
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">完成进度</Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <LinearProgress
                      variant="determinate"
                      value={selectedTask.progress}
                      sx={{ width: 200, mr: 2 }}
                    />
                    <Typography variant="body2">
                      {selectedTask.progress}%
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">质量评分</Typography>
                  <Typography variant="body2" color="success.main">
                    {(selectedTask.quality_score * 100).toFixed(1)}%
                  </Typography>
                </Grid>
              </Grid>

              <Typography variant="subtitle2" gutterBottom>
                分配人员
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                {selectedTask.assignees.map((assignee: string) => (
                  <Chip
                    key={assignee}
                    label={assignee}
                    variant="outlined"
                  />
                ))}
              </Box>

              <Typography variant="subtitle2" gutterBottom>
                时间信息
              </Typography>
              <Typography variant="body2">
                创建时间: {selectedTask.created_at}
              </Typography>
              <Typography variant="body2">
                截止时间: {selectedTask.deadline}
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setProgressDialog(false)}>关闭</Button>
          <Button variant="contained">生成报告</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}