/**
 * 实验配置表单组件
 */
import React, { useState, useCallback } from 'react';
import {
  Box,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Card,
  CardContent,
  Typography,
  Grid,
  Switch,
  FormControlLabel,
  Chip,
  IconButton,
  Slider,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  InputAdornment,
  Autocomplete,
  Stack,
  // Divider,  // 暂时注释未使用的导入
  Tooltip
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  ExpandMore as ExpandMoreIcon,
  Info as InfoIcon,
  // ContentCopy as CopyIcon,  // 暂时注释未使用的导入
  // Settings as SettingsIcon  // 暂时注释未使用的导入
} from '@mui/icons-material';

// 实验类型
export enum ExperimentType {
  AB = 'A/B',
  MULTIVARIATE = '多变体',
  HOLDOUT = '保留组',
  FEATURE_FLAG = '功能开关'
}

// 实验状态
export enum ExperimentStatus {
  DRAFT = '草稿',
  ACTIVE = '运行中',
  PAUSED = '已暂停',
  COMPLETED = '已完成',
  ARCHIVED = '已归档'
}

// 变体配置
interface Variant {
  id: string;
  name: string;
  description: string;
  traffic: number;
  isControl: boolean;
  config: Record<string, any>;
}

// 指标配置
interface Metric {
  id: string;
  name: string;
  type: 'primary' | 'secondary' | 'guardrail';
  aggregation: 'mean' | 'sum' | 'count' | 'conversion';
  unit?: string;
  minimumDetectableEffect?: number;
  expectedDirection?: 'increase' | 'decrease' | 'any';
}

// 定向规则
interface TargetingRule {
  id: string;
  type: 'user' | 'segment' | 'property';
  operator: 'equals' | 'contains' | 'in' | 'not_in' | 'greater_than' | 'less_than';
  field: string;
  value: any;
}

// 实验配置
export interface ExperimentConfig {
  name: string;
  description: string;
  type: ExperimentType;
  status: ExperimentStatus;
  startDate?: Date;
  endDate?: Date;
  variants: Variant[];
  metrics: Metric[];
  targetingRules: TargetingRule[];
  sampleSize?: number;
  confidenceLevel: number;
  mutuallyExclusiveGroup?: string;
  layer?: string;
  tags: string[];
  enableDataQualityChecks: boolean;
  enableAutoStop: boolean;
  autoStopThreshold?: number;
}

interface ExperimentConfigFormProps {
  initialConfig?: Partial<ExperimentConfig>;
  onSubmit: (config: ExperimentConfig) => void;
  onCancel?: () => void;
  mode?: 'create' | 'edit' | 'view';
}

const ExperimentConfigForm: React.FC<ExperimentConfigFormProps> = ({
  initialConfig,
  onSubmit,
  onCancel,
  mode = 'create'
}) => {
  const [config, setConfig] = useState<ExperimentConfig>({
    name: '',
    description: '',
    type: ExperimentType.AB,
    status: ExperimentStatus.DRAFT,
    variants: [
      { id: '1', name: '对照组', description: '', traffic: 50, isControl: true, config: {} },
      { id: '2', name: '实验组', description: '', traffic: 50, isControl: false, config: {} }
    ],
    metrics: [],
    targetingRules: [],
    confidenceLevel: 95,
    tags: [],
    enableDataQualityChecks: true,
    enableAutoStop: false,
    ...initialConfig
  });

  const [errors, setErrors] = useState<Record<string, string>>({});
  const [expandedSections, setExpandedSections] = useState<string[]>(['basic', 'variants', 'metrics']);

  // 更新配置
  const updateConfig = useCallback((field: keyof ExperimentConfig, value: any) => {
    setConfig(prev => ({ ...prev, [field]: value }));
    // 清除该字段的错误
    setErrors(prev => ({ ...prev, [field]: '' }));
  }, []);

  // 添加变体
  const addVariant = useCallback(() => {
    const newVariant: Variant = {
      id: Date.now().toString(),
      name: `变体${config.variants.length}`,
      description: '',
      traffic: 0,
      isControl: false,
      config: {}
    };
    updateConfig('variants', [...config.variants, newVariant]);
  }, [config.variants, updateConfig]);

  // 更新变体
  const updateVariant = useCallback((id: string, field: keyof Variant, value: any) => {
    const updatedVariants = config.variants.map(v =>
      v.id === id ? { ...v, [field]: value } : v
    );
    updateConfig('variants', updatedVariants);
  }, [config.variants, updateConfig]);

  // 删除变体
  const deleteVariant = useCallback((id: string) => {
    updateConfig('variants', config.variants.filter(v => v.id !== id));
  }, [config.variants, updateConfig]);

  // 自动调整流量分配
  const autoDistributeTraffic = useCallback(() => {
    const equalTraffic = Math.floor(100 / config.variants.length);
    const remainder = 100 - (equalTraffic * config.variants.length);
    const updatedVariants = config.variants.map((v, index) => ({
      ...v,
      traffic: equalTraffic + (index === 0 ? remainder : 0)
    }));
    updateConfig('variants', updatedVariants);
  }, [config.variants, updateConfig]);

  // 添加指标
  const addMetric = useCallback(() => {
    const newMetric: Metric = {
      id: Date.now().toString(),
      name: '',
      type: 'primary',
      aggregation: 'mean'
    };
    updateConfig('metrics', [...config.metrics, newMetric]);
  }, [config.metrics, updateConfig]);

  // 更新指标
  const updateMetric = useCallback((id: string, field: keyof Metric, value: any) => {
    const updatedMetrics = config.metrics.map(m =>
      m.id === id ? { ...m, [field]: value } : m
    );
    updateConfig('metrics', updatedMetrics);
  }, [config.metrics, updateConfig]);

  // 删除指标
  const deleteMetric = useCallback((id: string) => {
    updateConfig('metrics', config.metrics.filter(m => m.id !== id));
  }, [config.metrics, updateConfig]);

  // 添加定向规则
  const addTargetingRule = useCallback(() => {
    const newRule: TargetingRule = {
      id: Date.now().toString(),
      type: 'user',
      operator: 'equals',
      field: '',
      value: ''
    };
    updateConfig('targetingRules', [...config.targetingRules, newRule]);
  }, [config.targetingRules, updateConfig]);

  // 更新定向规则
  const updateTargetingRule = useCallback((id: string, field: keyof TargetingRule, value: any) => {
    const updatedRules = config.targetingRules.map(r =>
      r.id === id ? { ...r, [field]: value } : r
    );
    updateConfig('targetingRules', updatedRules);
  }, [config.targetingRules, updateConfig]);

  // 删除定向规则
  const deleteTargetingRule = useCallback((id: string) => {
    updateConfig('targetingRules', config.targetingRules.filter(r => r.id !== id));
  }, [config.targetingRules, updateConfig]);

  // 验证表单
  const validateForm = useCallback((): boolean => {
    const newErrors: Record<string, string> = {};

    if (!config.name.trim()) {
      newErrors.name = '实验名称不能为空';
    }

    if (config.variants.length < 2) {
      newErrors.variants = '至少需要2个变体';
    }

    const totalTraffic = config.variants.reduce((sum, v) => sum + v.traffic, 0);
    if (Math.abs(totalTraffic - 100) > 0.01) {
      newErrors.traffic = `流量分配总和必须为100%，当前为${totalTraffic}%`;
    }

    if (config.metrics.length === 0) {
      newErrors.metrics = '至少需要一个主要指标';
    }

    const primaryMetrics = config.metrics.filter(m => m.type === 'primary');
    if (primaryMetrics.length === 0) {
      newErrors.primaryMetric = '至少需要一个主要指标';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [config]);

  // 提交表单
  const handleSubmit = useCallback(() => {
    if (validateForm()) {
      onSubmit(config);
    }
  }, [config, validateForm, onSubmit]);

  // 展开/收起部分
  const toggleSection = useCallback((section: string) => {
    setExpandedSections(prev =>
      prev.includes(section)
        ? prev.filter(s => s !== section)
        : [...prev, section]
    );
  }, []);

  const isReadOnly = mode === 'view';

  return (
    <Box sx={{ p: 2 }}>
      {/* 基本信息 */}
      <Accordion
        expanded={expandedSections.includes('basic')}
        onChange={() => toggleSection('basic')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">基本信息</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid {...{ item: true, xs: 12, md: 6 } as any}>
              <TextField
                fullWidth
                label="实验名称"
                value={config.name}
                onChange={(e) => updateConfig('name', e.target.value)}
                error={!!errors.name}
                helperText={errors.name}
                disabled={isReadOnly}
                required
              />
            </Grid>
            <Grid {...{ item: true, xs: 12, md: 6 } as any}>
              <FormControl fullWidth>
                <InputLabel>实验类型</InputLabel>
                <Select
                  value={config.type}
                  onChange={(e) => updateConfig('type', e.target.value)}
                  disabled={isReadOnly}
                  label="实验类型"
                >
                  {Object.values(ExperimentType).map(type => (
                    <MenuItem key={type} value={type}>{type}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid {...{ item: true, xs: 12 } as any}>
              <TextField
                fullWidth
                multiline
                rows={3}
                label="实验描述"
                value={config.description}
                onChange={(e) => updateConfig('description', e.target.value)}
                disabled={isReadOnly}
              />
            </Grid>
            <Grid {...{ item: true, xs: 12, md: 6 } as any}>
              <TextField
                fullWidth
                type="date"
                label="开始日期"
                value={config.startDate?.toISOString().split('T')[0] || ''}
                onChange={(e) => updateConfig('startDate', new Date(e.target.value))}
                InputLabelProps={{ shrink: true }}
                disabled={isReadOnly}
              />
            </Grid>
            <Grid {...{ item: true, xs: 12, md: 6 } as any}>
              <TextField
                fullWidth
                type="date"
                label="结束日期"
                value={config.endDate?.toISOString().split('T')[0] || ''}
                onChange={(e) => updateConfig('endDate', new Date(e.target.value))}
                InputLabelProps={{ shrink: true }}
                disabled={isReadOnly}
              />
            </Grid>
            <Grid {...{ item: true, xs: 12 } as any}>
              <Autocomplete
                multiple
                freeSolo
                options={[]}
                value={config.tags}
                onChange={(_, newValue) => updateConfig('tags', newValue)}
                renderTags={(value, getTagProps) =>
                  value.map((option, index) => (
                    <Chip
                      label={option}
                      {...getTagProps({ index })}
                      disabled={isReadOnly}
                    />
                  ))
                }
                renderInput={(params) => (
                  <TextField
                    {...params}
                    label="标签"
                    placeholder="输入标签后按回车"
                    disabled={isReadOnly}
                  />
                )}
                disabled={isReadOnly}
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* 变体配置 */}
      <Accordion
        expanded={expandedSections.includes('variants')}
        onChange={() => toggleSection('variants')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">
            变体配置
            {errors.variants && (
              <Chip label={errors.variants} color="error" size="small" sx={{ ml: 1 }} />
            )}
            {errors.traffic && (
              <Chip label={errors.traffic} color="error" size="small" sx={{ ml: 1 }} />
            )}
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Stack spacing={2}>
            {config.variants.map((variant, index) => (
              <Card key={variant.id} variant="outlined">
                <CardContent>
                  <Grid container spacing={2} alignItems="center">
                    <Grid {...{ item: true, xs: 12, md: 3 } as any}>
                      <TextField
                        fullWidth
                        label="变体名称"
                        value={variant.name}
                        onChange={(e) => updateVariant(variant.id, 'name', e.target.value)}
                        disabled={isReadOnly}
                        InputProps={{
                          startAdornment: variant.isControl && (
                            <InputAdornment position="start">
                              <Chip label="对照组" size="small" color="primary" />
                            </InputAdornment>
                          )
                        }}
                      />
                    </Grid>
                    <Grid {...{ item: true, xs: 12, md: 4 } as any}>
                      <TextField
                        fullWidth
                        label="描述"
                        value={variant.description}
                        onChange={(e) => updateVariant(variant.id, 'description', e.target.value)}
                        disabled={isReadOnly}
                      />
                    </Grid>
                    <Grid {...{ item: true, xs: 12, md: 3 } as any}>
                      <Box sx={{ px: 2 }}>
                        <Typography gutterBottom>流量分配: {variant.traffic}%</Typography>
                        <Slider
                          value={variant.traffic}
                          onChange={(_, value) => updateVariant(variant.id, 'traffic', value as number)}
                          disabled={isReadOnly}
                          valueLabelDisplay="auto"
                          min={0}
                          max={100}
                        />
                      </Box>
                    </Grid>
                    <Grid {...{ item: true, xs: 12, md: 2 } as any}>
                      <Stack direction="row" spacing={1}>
                        {!variant.isControl && !isReadOnly && (
                          <FormControlLabel
                            control={
                              <Switch
                                checked={variant.isControl}
                                onChange={(e) => updateVariant(variant.id, 'isControl', e.target.checked)}
                              />
                            }
                            label="对照组"
                          />
                        )}
                        {!isReadOnly && config.variants.length > 2 && (
                          <IconButton
                            onClick={() => deleteVariant(variant.id)}
                            color="error"
                            disabled={variant.isControl}
                          >
                            <DeleteIcon />
                          </IconButton>
                        )}
                      </Stack>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            ))}
            {!isReadOnly && (
              <Stack direction="row" spacing={2}>
                <Button
                  startIcon={<AddIcon />}
                  onClick={addVariant}
                  variant="outlined"
                >
                  添加变体
                </Button>
                <Button
                  onClick={autoDistributeTraffic}
                  variant="outlined"
                >
                  平均分配流量
                </Button>
              </Stack>
            )}
          </Stack>
        </AccordionDetails>
      </Accordion>

      {/* 指标配置 */}
      <Accordion
        expanded={expandedSections.includes('metrics')}
        onChange={() => toggleSection('metrics')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">
            指标配置
            {(errors.metrics || errors.primaryMetric) && (
              <Chip 
                label={errors.metrics || errors.primaryMetric} 
                color="error" 
                size="small" 
                sx={{ ml: 1 }} 
              />
            )}
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Stack spacing={2}>
            {config.metrics.map((metric) => (
              <Card key={metric.id} variant="outlined">
                <CardContent>
                  <Grid container spacing={2} alignItems="center">
                    <Grid {...{ item: true, xs: 12, md: 3 } as any}>
                      <TextField
                        fullWidth
                        label="指标名称"
                        value={metric.name}
                        onChange={(e) => updateMetric(metric.id, 'name', e.target.value)}
                        disabled={isReadOnly}
                        required
                      />
                    </Grid>
                    <Grid {...{ item: true, xs: 12, md: 2 } as any}>
                      <FormControl fullWidth>
                        <InputLabel>类型</InputLabel>
                        <Select
                          value={metric.type}
                          onChange={(e) => updateMetric(metric.id, 'type', e.target.value)}
                          disabled={isReadOnly}
                          label="类型"
                        >
                          <MenuItem value="primary">主要指标</MenuItem>
                          <MenuItem value="secondary">次要指标</MenuItem>
                          <MenuItem value="guardrail">护栏指标</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid {...{ item: true, xs: 12, md: 2 } as any}>
                      <FormControl fullWidth>
                        <InputLabel>聚合方式</InputLabel>
                        <Select
                          value={metric.aggregation}
                          onChange={(e) => updateMetric(metric.id, 'aggregation', e.target.value)}
                          disabled={isReadOnly}
                          label="聚合方式"
                        >
                          <MenuItem value="mean">平均值</MenuItem>
                          <MenuItem value="sum">总和</MenuItem>
                          <MenuItem value="count">计数</MenuItem>
                          <MenuItem value="conversion">转化率</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid {...{ item: true, xs: 12, md: 2 } as any}>
                      <TextField
                        fullWidth
                        label="单位"
                        value={metric.unit || ''}
                        onChange={(e) => updateMetric(metric.id, 'unit', e.target.value)}
                        disabled={isReadOnly}
                        placeholder="如: ms, %"
                      />
                    </Grid>
                    <Grid {...{ item: true, xs: 12, md: 2 } as any}>
                      <TextField
                        fullWidth
                        label="MDE"
                        type="number"
                        value={metric.minimumDetectableEffect || ''}
                        onChange={(e) => updateMetric(metric.id, 'minimumDetectableEffect', parseFloat(e.target.value))}
                        disabled={isReadOnly}
                        InputProps={{
                          endAdornment: (
                            <Tooltip title="最小可检测效应">
                              <InfoIcon fontSize="small" color="action" />
                            </Tooltip>
                          )
                        }}
                      />
                    </Grid>
                    <Grid {...{ item: true, xs: 12, md: 1 } as any}>
                      {!isReadOnly && (
                        <IconButton
                          onClick={() => deleteMetric(metric.id)}
                          color="error"
                        >
                          <DeleteIcon />
                        </IconButton>
                      )}
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            ))}
            {!isReadOnly && (
              <Button
                startIcon={<AddIcon />}
                onClick={addMetric}
                variant="outlined"
              >
                添加指标
              </Button>
            )}
          </Stack>
        </AccordionDetails>
      </Accordion>

      {/* 定向规则 */}
      <Accordion
        expanded={expandedSections.includes('targeting')}
        onChange={() => toggleSection('targeting')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">定向规则（可选）</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Stack spacing={2}>
            {config.targetingRules.map((rule) => (
              <Card key={rule.id} variant="outlined">
                <CardContent>
                  <Grid container spacing={2} alignItems="center">
                    <Grid {...{ item: true, xs: 12, md: 2 } as any}>
                      <FormControl fullWidth>
                        <InputLabel>类型</InputLabel>
                        <Select
                          value={rule.type}
                          onChange={(e) => updateTargetingRule(rule.id, 'type', e.target.value)}
                          disabled={isReadOnly}
                          label="类型"
                        >
                          <MenuItem value="user">用户ID</MenuItem>
                          <MenuItem value="segment">用户分组</MenuItem>
                          <MenuItem value="property">属性</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid {...{ item: true, xs: 12, md: 3 } as any}>
                      <TextField
                        fullWidth
                        label="字段"
                        value={rule.field}
                        onChange={(e) => updateTargetingRule(rule.id, 'field', e.target.value)}
                        disabled={isReadOnly}
                        placeholder="如: user_id, country"
                      />
                    </Grid>
                    <Grid {...{ item: true, xs: 12, md: 2 } as any}>
                      <FormControl fullWidth>
                        <InputLabel>操作符</InputLabel>
                        <Select
                          value={rule.operator}
                          onChange={(e) => updateTargetingRule(rule.id, 'operator', e.target.value)}
                          disabled={isReadOnly}
                          label="操作符"
                        >
                          <MenuItem value="equals">等于</MenuItem>
                          <MenuItem value="contains">包含</MenuItem>
                          <MenuItem value="in">在列表中</MenuItem>
                          <MenuItem value="not_in">不在列表中</MenuItem>
                          <MenuItem value="greater_than">大于</MenuItem>
                          <MenuItem value="less_than">小于</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid {...{ item: true, xs: 12, md: 4 } as any}>
                      <TextField
                        fullWidth
                        label="值"
                        value={rule.value}
                        onChange={(e) => updateTargetingRule(rule.id, 'value', e.target.value)}
                        disabled={isReadOnly}
                        placeholder="输入匹配值"
                      />
                    </Grid>
                    <Grid {...{ item: true, xs: 12, md: 1 } as any}>
                      {!isReadOnly && (
                        <IconButton
                          onClick={() => deleteTargetingRule(rule.id)}
                          color="error"
                        >
                          <DeleteIcon />
                        </IconButton>
                      )}
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            ))}
            {!isReadOnly && (
              <Button
                startIcon={<AddIcon />}
                onClick={addTargetingRule}
                variant="outlined"
              >
                添加定向规则
              </Button>
            )}
          </Stack>
        </AccordionDetails>
      </Accordion>

      {/* 高级设置 */}
      <Accordion
        expanded={expandedSections.includes('advanced')}
        onChange={() => toggleSection('advanced')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">高级设置</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid {...{ item: true, xs: 12, md: 6 } as any}>
              <Box sx={{ px: 2 }}>
                <Typography gutterBottom>
                  置信水平: {config.confidenceLevel}%
                </Typography>
                <Slider
                  value={config.confidenceLevel}
                  onChange={(_, value) => updateConfig('confidenceLevel', value as number)}
                  disabled={isReadOnly}
                  valueLabelDisplay="auto"
                  min={80}
                  max={99}
                  marks={[
                    { value: 80, label: '80%' },
                    { value: 90, label: '90%' },
                    { value: 95, label: '95%' },
                    { value: 99, label: '99%' }
                  ]}
                />
              </Box>
            </Grid>
            <Grid {...{ item: true, xs: 12, md: 6 } as any}>
              <TextField
                fullWidth
                label="样本量"
                type="number"
                value={config.sampleSize || ''}
                onChange={(e) => updateConfig('sampleSize', parseInt(e.target.value))}
                disabled={isReadOnly}
                helperText="留空表示自动计算"
              />
            </Grid>
            <Grid {...{ item: true, xs: 12, md: 6 } as any}>
              <TextField
                fullWidth
                label="互斥组"
                value={config.mutuallyExclusiveGroup || ''}
                onChange={(e) => updateConfig('mutuallyExclusiveGroup', e.target.value)}
                disabled={isReadOnly}
                helperText="同一互斥组的实验不会同时运行"
              />
            </Grid>
            <Grid {...{ item: true, xs: 12, md: 6 } as any}>
              <TextField
                fullWidth
                label="实验层"
                value={config.layer || ''}
                onChange={(e) => updateConfig('layer', e.target.value)}
                disabled={isReadOnly}
                helperText="用于分层实验"
              />
            </Grid>
            <Grid {...{ item: true, xs: 12 } as any}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.enableDataQualityChecks}
                    onChange={(e) => updateConfig('enableDataQualityChecks', e.target.checked)}
                    disabled={isReadOnly}
                  />
                }
                label="启用数据质量检查"
              />
            </Grid>
            <Grid {...{ item: true, xs: 12 } as any}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.enableAutoStop}
                    onChange={(e) => updateConfig('enableAutoStop', e.target.checked)}
                    disabled={isReadOnly}
                  />
                }
                label="启用自动停止"
              />
            </Grid>
            {config.enableAutoStop && (
              <Grid {...{ item: true, xs: 12 } as any}>
                <TextField
                  fullWidth
                  label="自动停止阈值"
                  type="number"
                  value={config.autoStopThreshold || ''}
                  onChange={(e) => updateConfig('autoStopThreshold', parseFloat(e.target.value))}
                  disabled={isReadOnly}
                  helperText="当指标恶化超过此阈值时自动停止实验"
                  InputProps={{
                    endAdornment: <InputAdornment position="end">%</InputAdornment>
                  }}
                />
              </Grid>
            )}
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* 错误提示 */}
      {Object.keys(errors).length > 0 && (
        <Alert severity="error" sx={{ mt: 2 }}>
          请修正表单中的错误后再提交
        </Alert>
      )}

      {/* 操作按钮 */}
      {!isReadOnly && (
        <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
          {onCancel && (
            <Button onClick={onCancel} variant="outlined">
              取消
            </Button>
          )}
          <Button
            onClick={handleSubmit}
            variant="contained"
            color="primary"
          >
            {mode === 'create' ? '创建实验' : '保存修改'}
          </Button>
        </Box>
      )}
    </Box>
  );
};

export default ExperimentConfigForm;