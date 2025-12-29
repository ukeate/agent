import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Box, Card, CardContent, Typography, Button, Grid, LinearProgress, Chip, Alert } from '@mui/material'
import { Refresh as RefreshIcon, Assignment as AssignmentIcon } from '@mui/icons-material'

interface TaskRow {
  task_id: string
  name: string
  task_type: string
  status: string
  progress?: number
  total_records?: number
  completed_records?: number
  created_at?: string
  deadline?: string
}

export default function AnnotationTasksPage() {
  const [tasks, setTasks] = useState<TaskRow[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/training-data/annotation-tasks'))
      const data = await res.json()
      setTasks(Array.isArray(data?.tasks) ? data.tasks : [])
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setTasks([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h4">标注任务</Typography>
        <Button startIcon={<RefreshIcon />} onClick={load} disabled={loading}>
          刷新
        </Button>
      </Box>

      {error && <Alert severity="error">{error}</Alert>}

      <Grid container spacing={3}>
        {tasks.length === 0 ? (
          <Grid item xs={12}>
            <Alert severity="info">暂无任务，先通过 /api/v1/training-data/annotation-tasks 创建。</Alert>
          </Grid>
        ) : (
          tasks.map((task) => (
            <Grid item xs={12} md={6} key={task.task_id}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={1}>
                    <AssignmentIcon color="primary" sx={{ mr: 1 }} />
                    <Typography variant="h6">{task.name}</Typography>
                    <Chip label={task.status} size="small" sx={{ ml: 1 }} />
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    类型: {task.task_type}
                  </Typography>
                  {typeof task.progress === 'number' && (
                    <Box mt={2}>
                      <LinearProgress variant="determinate" value={task.progress} />
                      <Typography variant="caption" color="text.secondary">
                        进度 {task.progress}%
                      </Typography>
                    </Box>
                  )}
                  <Box display="flex" justifyContent="space-between" mt={2}>
                    <Typography variant="body2">总数 {task.total_records ?? '-'}</Typography>
                    <Typography variant="body2">已完成 {task.completed_records ?? '-'}</Typography>
                  </Box>
                  <Box display="flex" justifyContent="space-between" mt={1}>
                    <Typography variant="body2">创建 {task.created_at || '-'}</Typography>
                    <Typography variant="body2">截止 {task.deadline || '-'}</Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))
        )}
      </Grid>
    </Box>
  )
}
