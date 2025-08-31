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
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  IconButton,
  Tooltip,
  List,
  ListItem,
  ListItemText
} from '@mui/material';
import {
  Add as AddIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Delete as DeleteIcon,
  Compare as CompareIcon,
  Restore as RestoreIcon,
  Merge as MergeIcon,
  Folder as FolderIcon
} from '@mui/icons-material';

export default function DataVersionManagementPage() {
  const [versions, setVersions] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [createDialog, setCreateDialog] = useState(false);
  const [compareDialog, setCompareDialog] = useState(false);

  const mockVersions = [
    {
      version_id: 'v1.0.0',
      dataset_name: 'news_classification',
      description: '初始版本',
      record_count: 10000,
      size_mb: 45.2,
      created_by: 'admin',
      created_at: '2024-01-15',
      quality_score: 0.87
    },
    {
      version_id: 'v1.1.0', 
      dataset_name: 'news_classification',
      description: '添加体育新闻数据',
      record_count: 12500,
      size_mb: 58.7,
      created_by: 'admin',
      created_at: '2024-01-18',
      quality_score: 0.91
    },
    {
      version_id: 'v1.2.0',
      dataset_name: 'news_classification', 
      description: '质量优化和数据清洗',
      record_count: 11800,
      size_mb: 52.3,
      created_by: 'admin',
      created_at: '2024-01-20',
      quality_score: 0.94
    }
  ];

  const mockDatasets = [
    { name: 'news_classification', version_count: 3, total_records: 11800 },
    { name: 'sentiment_analysis', version_count: 2, total_records: 8500 },
    { name: 'qa_dataset', version_count: 1, total_records: 5000 }
  ];

  useEffect(() => {
    setVersions(mockVersions);
    setDatasets(mockDatasets);
    setSelectedDataset('news_classification');
  }, []);

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        数据版本管理
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        管理数据集版本，支持版本比较、回滚和合并操作
      </Typography>

      <Grid container spacing={3}>
        {/* 数据集列表 */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                数据集列表
              </Typography>
              <List>
                {mockDatasets.map((dataset) => (
                  <ListItem
                    key={dataset.name}
                    button
                    selected={selectedDataset === dataset.name}
                    onClick={() => setSelectedDataset(dataset.name)}
                  >
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <FolderIcon sx={{ mr: 1 }} />
                          {dataset.name}
                        </Box>
                      }
                      secondary={`${dataset.version_count} 个版本 | ${dataset.total_records} 条记录`}
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* 版本列表 */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  版本列表 - {selectedDataset}
                </Typography>
                <Box>
                  <Button
                    variant="outlined"
                    startIcon={<RefreshIcon />}
                    sx={{ mr: 2 }}
                  >
                    刷新
                  </Button>
                  <Button
                    variant="contained"
                    startIcon={<AddIcon />}
                    onClick={() => setCreateDialog(true)}
                  >
                    创建版本
                  </Button>
                </Box>
              </Box>

              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>版本</TableCell>
                      <TableCell>描述</TableCell>
                      <TableCell>记录数</TableCell>
                      <TableCell>大小</TableCell>
                      <TableCell>质量</TableCell>
                      <TableCell>创建时间</TableCell>
                      <TableCell>操作</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {mockVersions
                      .filter(v => v.dataset_name === selectedDataset)
                      .map((version) => (
                      <TableRow key={version.version_id} hover>
                        <TableCell>
                          <Chip
                            label={version.version_id}
                            color="primary"
                            variant="outlined"
                          />
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">
                            {version.description}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            by {version.created_by}
                          </Typography>
                        </TableCell>
                        <TableCell>{version.record_count.toLocaleString()}</TableCell>
                        <TableCell>{version.size_mb} MB</TableCell>
                        <TableCell>
                          <Chip
                            label={`${(version.quality_score * 100).toFixed(1)}%`}
                            color={version.quality_score >= 0.9 ? 'success' : 'warning'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>{version.created_at}</TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', gap: 0.5 }}>
                            <Tooltip title="下载">
                              <IconButton size="small" color="primary">
                                <DownloadIcon />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="比较">
                              <IconButton 
                                size="small" 
                                color="info"
                                onClick={() => setCompareDialog(true)}
                              >
                                <CompareIcon />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="回滚">
                              <IconButton size="small" color="warning">
                                <RestoreIcon />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="删除">
                              <IconButton size="small" color="error">
                                <DeleteIcon />
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
        </Grid>
      </Grid>

      {/* 创建版本对话框 */}
      <Dialog open={createDialog} onClose={() => setCreateDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>创建新版本</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            margin="normal"
            label="版本号"
            placeholder="例如: v1.3.0"
          />
          <TextField
            fullWidth
            margin="normal"
            label="版本描述"
            multiline
            rows={3}
            placeholder="描述这个版本的主要变更"
          />
          <TextField
            fullWidth
            margin="normal"
            label="数据过滤条件"
            placeholder="JSON格式的过滤条件"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialog(false)}>取消</Button>
          <Button variant="contained">创建版本</Button>
        </DialogActions>
      </Dialog>

      {/* 版本比较对话框 */}
      <Dialog open={compareDialog} onClose={() => setCompareDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>版本比较</DialogTitle>
        <DialogContent>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Typography variant="h6" gutterBottom>版本 A: v1.1.0</Typography>
              <Typography variant="body2">记录数: 12,500</Typography>
              <Typography variant="body2">质量分: 91%</Typography>
              <Typography variant="body2">大小: 58.7 MB</Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="h6" gutterBottom>版本 B: v1.2.0</Typography>
              <Typography variant="body2">记录数: 11,800</Typography>
              <Typography variant="body2">质量分: 94%</Typography>
              <Typography variant="body2">大小: 52.3 MB</Typography>
            </Grid>
          </Grid>
          <Typography variant="h6" sx={{ mt: 2, mb: 1 }}>变更摘要</Typography>
          <Typography variant="body2">• 删除了700条低质量记录</Typography>
          <Typography variant="body2">• 质量评分提升了3%</Typography>
          <Typography variant="body2">• 文件大小减少了6.4 MB</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCompareDialog(false)}>关闭</Button>
          <Button variant="contained" startIcon={<MergeIcon />}>
            合并版本
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}