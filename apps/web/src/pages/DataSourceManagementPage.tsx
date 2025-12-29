import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState, useEffect } from 'react';
import {
import { logger } from '../utils/logger'
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Alert,
  CircularProgress,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  Add as AddIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Settings as SettingsIcon,
  CloudDownload as CloudDownloadIcon,
  Storage as DatabaseIcon,
  Language as WebIcon,
  InsertDriveFile as FileIcon
} from '@mui/icons-material';

interface DataSource {
  id: string;
  source_id: string;
  source_type: string;
  name: string;
  description: string;
  config: any;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export default function DataSourceManagementPage() {
  const [dataSources, setDataSources] = useState<DataSource[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [createDialog, setCreateDialog] = useState(false);
  const [editDialog, setEditDialog] = useState(false);
  const [selectedSource, setSelectedSource] = useState<DataSource | null>(null);
  const [newDataSource, setNewDataSource] = useState({
    source_id: '',
    source_type: 'file',
    name: '',
    description: '',
    config: {}
  });
  const [statistics, setStatistics] = useState<any>(null);

  const fetchDataSources = async () => {
    try {
      setLoading(true);
      const response = await apiFetch(buildApiUrl('/api/v1/training-data/sources'));
      const data = await response.json();
      setDataSources(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : '获取数据源失败');
    } finally {
      setLoading(false);
    }
  };

  const fetchStatistics = async () => {
    try {
      const response = await apiFetch(buildApiUrl('/api/v1/training-data/stats/overview'));
      const data = await response.json();
      setStatistics(data);
    } catch (err) {
      logger.error('获取统计信息失败:', err);
    }
  };

  const handleCreateDataSource = async () => {
    try {
      setLoading(true);
      const response = await apiFetch(buildApiUrl('/api/v1/training-data/sources'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newDataSource)
      });
      
      await response.json().catch(() => null);
      setCreateDialog(false);
      setNewDataSource({
        source_id: '',
        source_type: 'file',
        name: '',
        description: '',
        config: {}
      });
      await fetchDataSources();
    } catch (err) {
      setError(err instanceof Error ? err.message : '创建数据源失败');
    } finally {
      setLoading(false);
    }
  };

  const handleCollectData = async (sourceId: string) => {
    try {
      setLoading(true);
      const response = await apiFetch(buildApiUrl(`/api/v1/training-data/collect/${sourceId}`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          processing_rules: ['text_cleaning', 'format_standardization', 'quality_filtering'],
          batch_size: 100
        })
      });
      
      await response.json().catch(() => null);
      alert('数据收集任务已启动');
    } catch (err) {
      setError(err instanceof Error ? err.message : '启动数据收集失败');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteSource = async (sourceId: string) => {
    if (!window.confirm('确定要删除这个数据源吗？')) {
      return;
    }

    try {
      setLoading(true);
      const response = await apiFetch(buildApiUrl(`/api/v1/training-data/sources/${sourceId}`), {
        method: 'DELETE'
      });
      
      await response.json().catch(() => null);
      await fetchDataSources();
    } catch (err) {
      setError(err instanceof Error ? err.message : '删除数据源失败');
    } finally {
      setLoading(false);
    }
  };

  const getSourceTypeIcon = (type: string) => {
    switch (type) {
      case 'api': return <CloudDownloadIcon />;
      case 'database': return <DatabaseIcon />;
      case 'web': return <WebIcon />;
      default: return <FileIcon />;
    }
  };

  const getSourceTypeColor = (type: string) => {
    switch (type) {
      case 'api': return 'primary';
      case 'database': return 'success';
      case 'web': return 'warning';
      default: return 'default';
    }
  };

  useEffect(() => {
    fetchDataSources();
    fetchStatistics();
  }, []);

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        数据源管理
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        管理和配置训练数据的来源，支持API、文件、数据库、Web爬虫等多种数据源类型
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* 统计概览 */}
      {statistics && (
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <DatabaseIcon color="primary" sx={{ mr: 2 }} />
                  <Box>
                    <Typography variant="h6" color="primary">
                      {statistics.sources?.total || 0}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      总数据源
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
                  <CloudDownloadIcon color="success" sx={{ mr: 2 }} />
                  <Box>
                    <Typography variant="h6" color="success.main">
                      {statistics.sources?.active || 0}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      活跃数据源
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
                  <FileIcon color="info" sx={{ mr: 2 }} />
                  <Box>
                    <Typography variant="h6" color="info.main">
                      {statistics.records?.total || 0}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      数据记录总数
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
                  <SettingsIcon color="warning" sx={{ mr: 2 }} />
                  <Box>
                    <Typography variant="h6" color="warning.main">
                      {((statistics.records?.processing_rate || 0) * 100).toFixed(1)}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      处理完成率
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
            onClick={fetchDataSources}
            disabled={loading}
            sx={{ mr: 2 }}
          >
            刷新
          </Button>
        </Box>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setCreateDialog(true)}
          disabled={loading}
        >
          添加数据源
        </Button>
      </Box>

      {/* 数据源列表 */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            数据源列表
          </Typography>
          <TableContainer component={Paper} variant="outlined">
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>数据源信息</TableCell>
                  <TableCell>类型</TableCell>
                  <TableCell>状态</TableCell>
                  <TableCell>创建时间</TableCell>
                  <TableCell>最后更新</TableCell>
                  <TableCell>操作</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {loading ? (
                  <TableRow>
                    <TableCell colSpan={6} align="center">
                      <CircularProgress />
                    </TableCell>
                  </TableRow>
                ) : dataSources.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={6} align="center">
                      <Typography variant="body2" color="text.secondary">
                        暂无数据源，点击"添加数据源"开始配置
                      </Typography>
                    </TableCell>
                  </TableRow>
                ) : (
                  dataSources.map((source) => (
                    <TableRow key={source.id} hover>
                      <TableCell>
                        <Box>
                          <Typography variant="subtitle2" fontWeight="bold">
                            {source.name}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            ID: {source.source_id}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {source.description}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip
                          icon={getSourceTypeIcon(source.source_type)}
                          label={source.source_type.toUpperCase()}
                          color={getSourceTypeColor(source.source_type) as any}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={source.is_active ? '活跃' : '非活跃'}
                          color={source.is_active ? 'success' : 'default'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {new Date(source.created_at).toLocaleString()}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {new Date(source.updated_at).toLocaleString()}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          <Tooltip title="收集数据">
                            <IconButton
                              size="small"
                              color="primary"
                              onClick={() => handleCollectData(source.source_id)}
                              disabled={!source.is_active || loading}
                            >
                              <DownloadIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="编辑配置">
                            <IconButton
                              size="small"
                              color="default"
                              onClick={() => {
                                setSelectedSource(source);
                                setEditDialog(true);
                              }}
                            >
                              <EditIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="删除数据源">
                            <IconButton
                              size="small"
                              color="error"
                              onClick={() => handleDeleteSource(source.source_id)}
                              disabled={loading}
                            >
                              <DeleteIcon />
                            </IconButton>
                          </Tooltip>
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

      {/* 创建数据源对话框 */}
      <Dialog open={createDialog} onClose={() => setCreateDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>创建数据源</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            margin="normal"
            label="数据源ID"
            value={newDataSource.source_id}
            onChange={(e) => setNewDataSource(prev => ({ ...prev, source_id: e.target.value }))}
            helperText="唯一标识符，只能包含字母、数字、下划线"
          />
          <FormControl fullWidth margin="normal">
            <InputLabel>数据源类型</InputLabel>
            <Select
              value={newDataSource.source_type}
              label="数据源类型"
              onChange={(e) => setNewDataSource(prev => ({ ...prev, source_type: e.target.value }))}
            >
              <MenuItem value="file">文件 - 本地或远程文件</MenuItem>
              <MenuItem value="api">API - REST API接口</MenuItem>
              <MenuItem value="web">网页 - 网页爬虫</MenuItem>
              <MenuItem value="database">数据库 - SQL数据库</MenuItem>
            </Select>
          </FormControl>
          <TextField
            fullWidth
            margin="normal"
            label="数据源名称"
            value={newDataSource.name}
            onChange={(e) => setNewDataSource(prev => ({ ...prev, name: e.target.value }))}
            helperText="便于识别的显示名称"
          />
          <TextField
            fullWidth
            margin="normal"
            label="描述"
            multiline
            rows={3}
            value={newDataSource.description}
            onChange={(e) => setNewDataSource(prev => ({ ...prev, description: e.target.value }))}
            helperText="详细说明数据源的用途和内容"
          />
          <TextField
            fullWidth
            margin="normal"
            label="配置 (JSON格式)"
            multiline
            rows={6}
            value={JSON.stringify(newDataSource.config, null, 2)}
            onChange={(e) => {
              try {
                const config = JSON.parse(e.target.value);
                setNewDataSource(prev => ({ ...prev, config }));
              } catch {
                // 忽略无效JSON
              }
            }}
            helperText="根据数据源类型配置相应的参数，如API地址、文件路径等"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialog(false)}>取消</Button>
          <Button 
            onClick={handleCreateDataSource} 
            variant="contained"
            disabled={loading || !newDataSource.source_id || !newDataSource.name}
          >
            {loading ? <CircularProgress size={20} /> : '创建'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* 编辑数据源对话框 */}
      <Dialog open={editDialog} onClose={() => setEditDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>编辑数据源</DialogTitle>
        <DialogContent>
          {selectedSource && (
            <>
              <TextField
                fullWidth
                margin="normal"
                label="数据源名称"
                defaultValue={selectedSource.name}
                disabled
              />
              <TextField
                fullWidth
                margin="normal"
                label="描述"
                multiline
                rows={3}
                defaultValue={selectedSource.description}
              />
              <TextField
                fullWidth
                margin="normal"
                label="配置 (JSON格式)"
                multiline
                rows={8}
                defaultValue={JSON.stringify(selectedSource.config, null, 2)}
                helperText="修改数据源的配置参数"
              />
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialog(false)}>取消</Button>
          <Button variant="contained" disabled={loading}>
            {loading ? <CircularProgress size={20} /> : '保存'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
