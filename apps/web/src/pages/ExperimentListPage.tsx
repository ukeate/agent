/**
 * 实验列表页面
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Container,
  Typography,
  Button,
  Grid,
  Paper,
  TextField,
  InputAdornment,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Stack,
  Tabs,
  Tab,
  IconButton,
  Menu,
  ListItemIcon,
  ListItemText,
  Pagination,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  DialogContentText,
  Skeleton,
  Badge,
  Tooltip,
  ToggleButton,
  ToggleButtonGroup
} from '@mui/material';
import {
  Add as AddIcon,
  Search as SearchIcon,
  FilterList as FilterIcon,
  ViewList as ViewListIcon,
  ViewModule as ViewModuleIcon,
  Download as DownloadIcon,
  Upload as UploadIcon,
  Settings as SettingsIcon,
  Archive as ArchiveIcon,
  Delete as DeleteIcon,
  ContentCopy as CopyIcon,
  Refresh as RefreshIcon,
  MoreVert as MoreVertIcon
} from '@mui/icons-material';
import ExperimentCard, { ExperimentData } from '../components/experiment/ExperimentCard';
import ExperimentConfigForm from '../components/experiment/ExperimentConfigForm';
import { experimentService } from '../services/experimentService';

// 过滤选项
interface FilterOptions {
  status?: string;
  type?: string;
  owner?: string;
  dateRange?: {
    start?: Date;
    end?: Date;
  };
  tags?: string[];
}

// 排序选项
type SortBy = 'name' | 'created' | 'updated' | 'status' | 'progress';
type SortOrder = 'asc' | 'desc';

const ExperimentListPage: React.FC = () => {
  // 状态管理
  const [experiments, setExperiments] = useState<ExperimentData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTab, setSelectedTab] = useState(0);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const [filters, setFilters] = useState<FilterOptions>({});
  const [sortBy, setSortBy] = useState<SortBy>('updated');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');
  const [page, setPage] = useState(1);
  const [pageSize] = useState(12);
  const [totalPages, setTotalPages] = useState(1);
  const [selectedExperiments, setSelectedExperiments] = useState<Set<string>>(new Set());
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [experimentToDelete, setExperimentToDelete] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  // 加载实验数据
  const loadExperiments = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await experimentService.listExperiments({
        search: searchQuery,
        ...filters,
        sortBy,
        sortOrder,
        page,
        pageSize
      });
      setExperiments(response.experiments);
      setTotalPages(Math.ceil(response.total / pageSize));
    } catch (err) {
      setError('加载实验列表失败');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [searchQuery, filters, sortBy, sortOrder, page, pageSize]);

  // 初始加载和刷新
  useEffect(() => {
    loadExperiments();
  }, [loadExperiments, refreshKey]);

  // 处理实验操作
  const handleExperimentAction = useCallback(async (action: string, experimentId: string) => {
    try {
      switch (action) {
        case 'start':
          await experimentService.startExperiment(experimentId);
          break;
        case 'pause':
          await experimentService.pauseExperiment(experimentId);
          break;
        case 'resume':
          await experimentService.resumeExperiment(experimentId);
          break;
        case 'stop':
          await experimentService.stopExperiment(experimentId);
          break;
        case 'clone':
          await experimentService.cloneExperiment(experimentId);
          break;
        case 'delete':
          setExperimentToDelete(experimentId);
          setShowDeleteDialog(true);
          return;
        case 'edit':
          // 导航到编辑页面
          window.location.href = `/experiments/${experimentId}/edit`;
          return;
        case 'report':
          // 导航到报告页面
          window.location.href = `/experiments/${experimentId}/report`;
          return;
      }
      // 刷新列表
      setRefreshKey(prev => prev + 1);
    } catch (err) {
      setError(`操作失败: ${err}`);
    }
  }, []);

  // 确认删除
  const confirmDelete = useCallback(async () => {
    if (experimentToDelete) {
      try {
        await experimentService.deleteExperiment(experimentToDelete);
        setRefreshKey(prev => prev + 1);
      } catch (err) {
        setError(`删除失败: ${err}`);
      }
    }
    setShowDeleteDialog(false);
    setExperimentToDelete(null);
  }, [experimentToDelete]);

  // 批量操作
  const handleBatchAction = useCallback(async (action: string) => {
    try {
      const ids = Array.from(selectedExperiments);
      switch (action) {
        case 'archive':
          await Promise.all(ids.map(id => experimentService.archiveExperiment(id)));
          break;
        case 'delete':
          await Promise.all(ids.map(id => experimentService.deleteExperiment(id)));
          break;
        case 'export':
          await experimentService.exportExperiments(ids);
          break;
      }
      setSelectedExperiments(new Set());
      setRefreshKey(prev => prev + 1);
    } catch (err) {
      setError(`批量操作失败: ${err}`);
    }
  }, [selectedExperiments]);

  // 获取过滤后的实验
  const getFilteredExperiments = useCallback(() => {
    let filtered = experiments;

    // 根据标签页过滤
    switch (selectedTab) {
      case 1: // 运行中
        filtered = filtered.filter(e => e.status === 'active');
        break;
      case 2: // 草稿
        filtered = filtered.filter(e => e.status === 'draft');
        break;
      case 3: // 已完成
        filtered = filtered.filter(e => e.status === 'completed');
        break;
      case 4: // 已归档
        filtered = filtered.filter(e => e.status === 'archived');
        break;
    }

    return filtered;
  }, [experiments, selectedTab]);

  // 统计数据
  const stats = {
    total: experiments.length,
    active: experiments.filter(e => e.status === 'active').length,
    draft: experiments.filter(e => e.status === 'draft').length,
    completed: experiments.filter(e => e.status === 'completed').length,
    archived: experiments.filter(e => e.status === 'archived').length
  };

  return (
    <Container maxWidth="xl" sx={{ py: 3 }}>
      {/* 页面标题和操作 */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1">
          实验管理
        </Typography>
        <Stack direction="row" spacing={2}>
          <Button
            variant="outlined"
            startIcon={<UploadIcon />}
          >
            导入
          </Button>
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            disabled={selectedExperiments.size === 0}
            onClick={() => handleBatchAction('export')}
          >
            导出 ({selectedExperiments.size})
          </Button>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setShowCreateDialog(true)}
          >
            创建实验
          </Button>
        </Stack>
      </Box>

      {/* 统计卡片 */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h4">{stats.total}</Typography>
            <Typography variant="body2" color="text.secondary">全部实验</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 2, textAlign: 'center', borderLeft: 4, borderColor: 'success.main' }}>
            <Typography variant="h4" color="success.main">{stats.active}</Typography>
            <Typography variant="body2" color="text.secondary">运行中</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 2, textAlign: 'center', borderLeft: 4, borderColor: 'warning.main' }}>
            <Typography variant="h4" color="warning.main">{stats.draft}</Typography>
            <Typography variant="body2" color="text.secondary">草稿</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 2, textAlign: 'center', borderLeft: 4, borderColor: 'info.main' }}>
            <Typography variant="h4" color="info.main">{stats.completed}</Typography>
            <Typography variant="body2" color="text.secondary">已完成</Typography>
          </Paper>
        </Grid>
      </Grid>

      {/* 搜索和过滤栏 */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              placeholder="搜索实验..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon />
                  </InputAdornment>
                )
              }}
            />
          </Grid>
          <Grid item xs={12} md={2}>
            <FormControl fullWidth>
              <InputLabel>类型</InputLabel>
              <Select
                value={filters.type || ''}
                onChange={(e) => setFilters({ ...filters, type: e.target.value })}
                label="类型"
              >
                <MenuItem value="">全部</MenuItem>
                <MenuItem value="A/B">A/B测试</MenuItem>
                <MenuItem value="多变体">多变体</MenuItem>
                <MenuItem value="功能开关">功能开关</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={2}>
            <FormControl fullWidth>
              <InputLabel>排序</InputLabel>
              <Select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as SortBy)}
                label="排序"
              >
                <MenuItem value="name">名称</MenuItem>
                <MenuItem value="created">创建时间</MenuItem>
                <MenuItem value="updated">更新时间</MenuItem>
                <MenuItem value="status">状态</MenuItem>
                <MenuItem value="progress">进度</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={2}>
            <ToggleButtonGroup
              value={sortOrder}
              exclusive
              onChange={(_, value) => value && setSortOrder(value)}
              fullWidth
            >
              <ToggleButton value="asc">升序</ToggleButton>
              <ToggleButton value="desc">降序</ToggleButton>
            </ToggleButtonGroup>
          </Grid>
          <Grid item xs={12} md={2}>
            <Stack direction="row" spacing={1} justifyContent="flex-end">
              <ToggleButtonGroup
                value={viewMode}
                exclusive
                onChange={(_, value) => value && setViewMode(value)}
              >
                <ToggleButton value="grid">
                  <ViewModuleIcon />
                </ToggleButton>
                <ToggleButton value="list">
                  <ViewListIcon />
                </ToggleButton>
              </ToggleButtonGroup>
              <IconButton onClick={() => setRefreshKey(prev => prev + 1)}>
                <RefreshIcon />
              </IconButton>
              <IconButton onClick={(e) => setAnchorEl(e.currentTarget)}>
                <MoreVertIcon />
              </IconButton>
            </Stack>
          </Grid>
        </Grid>
      </Paper>

      {/* 标签页 */}
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={selectedTab}
          onChange={(_, value) => setSelectedTab(value)}
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab 
            label={
              <Badge badgeContent={stats.total} color="default">
                全部
              </Badge>
            } 
          />
          <Tab 
            label={
              <Badge badgeContent={stats.active} color="success">
                运行中
              </Badge>
            }
          />
          <Tab 
            label={
              <Badge badgeContent={stats.draft} color="warning">
                草稿
              </Badge>
            }
          />
          <Tab 
            label={
              <Badge badgeContent={stats.completed} color="info">
                已完成
              </Badge>
            }
          />
          <Tab 
            label={
              <Badge badgeContent={stats.archived} color="default">
                已归档
              </Badge>
            }
          />
        </Tabs>
      </Paper>

      {/* 错误提示 */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* 实验列表 */}
      {loading ? (
        <Grid container spacing={3}>
          {[...Array(6)].map((_, index) => (
            <Grid item xs={12} md={viewMode === 'grid' ? 4 : 12} key={index}>
              <Skeleton variant="rectangular" height={300} />
            </Grid>
          ))}
        </Grid>
      ) : (
        <>
          <Grid container spacing={3}>
            {getFilteredExperiments().map((experiment) => (
              <Grid 
                item 
                xs={12} 
                md={viewMode === 'grid' ? 4 : 12} 
                key={experiment.id}
              >
                <ExperimentCard
                  experiment={experiment}
                  onAction={handleExperimentAction}
                  showDetails={viewMode === 'list'}
                  compact={viewMode === 'grid'}
                />
              </Grid>
            ))}
          </Grid>

          {/* 分页 */}
          {totalPages > 1 && (
            <Box sx={{ mt: 4, display: 'flex', justifyContent: 'center' }}>
              <Pagination
                count={totalPages}
                page={page}
                onChange={(_, value) => setPage(value)}
                color="primary"
                size="large"
              />
            </Box>
          )}

          {/* 空状态 */}
          {getFilteredExperiments().length === 0 && (
            <Box sx={{ textAlign: 'center', py: 8 }}>
              <Typography variant="h6" color="text.secondary" gutterBottom>
                没有找到实验
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                {searchQuery ? '尝试调整搜索条件' : '点击"创建实验"开始'}
              </Typography>
              {!searchQuery && (
                <Button
                  variant="contained"
                  startIcon={<AddIcon />}
                  onClick={() => setShowCreateDialog(true)}
                >
                  创建第一个实验
                </Button>
              )}
            </Box>
          )}
        </>
      )}

      {/* 批量操作菜单 */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={() => setAnchorEl(null)}
      >
        <MenuItem 
          onClick={() => {
            handleBatchAction('archive');
            setAnchorEl(null);
          }}
          disabled={selectedExperiments.size === 0}
        >
          <ListItemIcon>
            <ArchiveIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>批量归档</ListItemText>
        </MenuItem>
        <MenuItem 
          onClick={() => {
            handleBatchAction('delete');
            setAnchorEl(null);
          }}
          disabled={selectedExperiments.size === 0}
        >
          <ListItemIcon>
            <DeleteIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>批量删除</ListItemText>
        </MenuItem>
      </Menu>

      {/* 创建实验对话框 */}
      <Dialog
        open={showCreateDialog}
        onClose={() => setShowCreateDialog(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>创建新实验</DialogTitle>
        <DialogContent>
          <ExperimentConfigForm
            onSubmit={async (config) => {
              try {
                await experimentService.createExperiment(config);
                setShowCreateDialog(false);
                setRefreshKey(prev => prev + 1);
              } catch (err) {
                setError(`创建失败: ${err}`);
              }
            }}
            onCancel={() => setShowCreateDialog(false)}
          />
        </DialogContent>
      </Dialog>

      {/* 删除确认对话框 */}
      <Dialog
        open={showDeleteDialog}
        onClose={() => setShowDeleteDialog(false)}
      >
        <DialogTitle>确认删除</DialogTitle>
        <DialogContent>
          <DialogContentText>
            确定要删除这个实验吗？此操作不可恢复。
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowDeleteDialog(false)}>取消</Button>
          <Button onClick={confirmDelete} color="error" variant="contained">
            删除
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default ExperimentListPage;