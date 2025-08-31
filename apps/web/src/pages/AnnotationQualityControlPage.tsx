import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Paper,
  LinearProgress,
  Chip,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import {
  Assessment as AssessmentIcon,
  TrendingUp as TrendingUpIcon,
  GroupWork as GroupWorkIcon,
  Speed as SpeedIcon,
  Star as StarIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';

export default function AnnotationQualityControlPage() {
  const [qualityReport, setQualityReport] = useState<any>(null);
  const [reportDialog, setReportDialog] = useState(false);
  const [selectedTask, setSelectedTask] = useState<any>(null);

  const mockQualityData = {
    overall_score: 0.89,
    agreement_metrics: {
      fleiss_kappa: 0.78,
      cohens_kappa: 0.82,
      percentage_agreement: 0.85
    },
    consistency_metrics: {
      avg_label_consistency: 0.87,
      avg_confidence_consistency: 0.91
    },
    annotator_performance: [
      {
        annotator_id: 'ann_001',
        name: '张小明',
        total_annotations: 1250,
        avg_confidence: 0.89,
        quality_score: 0.92,
        consistency_score: 0.88
      },
      {
        annotator_id: 'ann_002',
        name: '李小红',
        total_annotations: 980,
        avg_confidence: 0.85,
        quality_score: 0.88,
        consistency_score: 0.85
      },
      {
        annotator_id: 'ann_003',
        name: '王大伟',
        total_annotations: 750,
        avg_confidence: 0.82,
        quality_score: 0.84,
        consistency_score: 0.82
      }
    ],
    conflict_analysis: {
      total_conflicts: 45,
      resolved_conflicts: 38,
      pending_conflicts: 7,
      conflict_rate: 0.023
    },
    recommendations: [
      '建议对质量评分低于85%的标注者进行额外培训',
      '存在7个冲突标注需要专家审核',
      '张小明表现优秀，可考虑担任质量审核员',
      '建议优化标注指南，减少歧义性描述'
    ]
  };

  useEffect(() => {
    setQualityReport(mockQualityData);
  }, []);

  const getScoreColor = (score: number) => {
    if (score >= 0.9) return 'success';
    if (score >= 0.8) return 'warning';
    return 'error';
  };

  const getScoreLabel = (score: number) => {
    if (score >= 0.9) return '优秀';
    if (score >= 0.8) return '良好';
    if (score >= 0.7) return '一般';
    return '需改进';
  };

  if (!qualityReport) {
    return <Box sx={{ p: 3 }}>加载中...</Box>;
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        标注质量控制
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        监控和分析标注质量，提供一致性检查和改进建议
      </Typography>

      {/* 质量概览 */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <StarIcon color="primary" sx={{ mr: 2 }} />
                <Box>
                  <Typography variant="h6" color="primary">
                    {(qualityReport.overall_score * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    整体质量评分
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
                <GroupWorkIcon color="success" sx={{ mr: 2 }} />
                <Box>
                  <Typography variant="h6" color="success.main">
                    {(qualityReport.agreement_metrics.fleiss_kappa * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    标注者一致性
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
                <WarningIcon color="warning" sx={{ mr: 2 }} />
                <Box>
                  <Typography variant="h6" color="warning.main">
                    {qualityReport.conflict_analysis.pending_conflicts}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    待解决冲突
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
                <TrendingUpIcon color="info" sx={{ mr: 2 }} />
                <Box>
                  <Typography variant="h6" color="info.main">
                    {(qualityReport.consistency_metrics.avg_label_consistency * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    内部一致性
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* 一致性分析 */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">一致性分析</Typography>
                <Button
                  variant="outlined"
                  startIcon={<RefreshIcon />}
                  size="small"
                >
                  刷新
                </Button>
              </Box>
              
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Fleiss' Kappa 系数
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <LinearProgress
                    variant="determinate"
                    value={qualityReport.agreement_metrics.fleiss_kappa * 100}
                    sx={{ width: '100%', mr: 2 }}
                    color={getScoreColor(qualityReport.agreement_metrics.fleiss_kappa) as any}
                  />
                  <Typography variant="body2">
                    {(qualityReport.agreement_metrics.fleiss_kappa * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <Typography variant="caption" color="text.secondary">
                  {getScoreLabel(qualityReport.agreement_metrics.fleiss_kappa)}
                </Typography>
              </Box>

              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Cohen's Kappa 系数
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <LinearProgress
                    variant="determinate"
                    value={qualityReport.agreement_metrics.cohens_kappa * 100}
                    sx={{ width: '100%', mr: 2 }}
                    color={getScoreColor(qualityReport.agreement_metrics.cohens_kappa) as any}
                  />
                  <Typography variant="body2">
                    {(qualityReport.agreement_metrics.cohens_kappa * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <Typography variant="caption" color="text.secondary">
                  {getScoreLabel(qualityReport.agreement_metrics.cohens_kappa)}
                </Typography>
              </Box>

              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  百分比一致性
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <LinearProgress
                    variant="determinate"
                    value={qualityReport.agreement_metrics.percentage_agreement * 100}
                    sx={{ width: '100%', mr: 2 }}
                    color={getScoreColor(qualityReport.agreement_metrics.percentage_agreement) as any}
                  />
                  <Typography variant="body2">
                    {(qualityReport.agreement_metrics.percentage_agreement * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <Typography variant="caption" color="text.secondary">
                  {getScoreLabel(qualityReport.agreement_metrics.percentage_agreement)}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* 冲突分析 */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                冲突分析
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2">
                  冲突统计
                </Typography>
                <Grid container spacing={2} sx={{ mt: 1 }}>
                  <Grid item xs={4}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h6" color="error.main">
                        {qualityReport.conflict_analysis.total_conflicts}
                      </Typography>
                      <Typography variant="caption">
                        总冲突
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={4}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h6" color="success.main">
                        {qualityReport.conflict_analysis.resolved_conflicts}
                      </Typography>
                      <Typography variant="caption">
                        已解决
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={4}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h6" color="warning.main">
                        {qualityReport.conflict_analysis.pending_conflicts}
                      </Typography>
                      <Typography variant="caption">
                        待处理
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Box>

              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  冲突率趋势
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <LinearProgress
                    variant="determinate"
                    value={qualityReport.conflict_analysis.conflict_rate * 100}
                    sx={{ width: '100%', mr: 2 }}
                    color={qualityReport.conflict_analysis.conflict_rate < 0.05 ? 'success' : 'warning'}
                  />
                  <Typography variant="body2">
                    {(qualityReport.conflict_analysis.conflict_rate * 100).toFixed(2)}%
                  </Typography>
                </Box>
                <Typography variant="caption" color="text.secondary">
                  {qualityReport.conflict_analysis.conflict_rate < 0.05 ? '冲突率正常' : '冲突率偏高'}
                </Typography>
              </Box>

              {qualityReport.conflict_analysis.pending_conflicts > 0 && (
                <Alert severity="warning" sx={{ mt: 2 }}>
                  有 {qualityReport.conflict_analysis.pending_conflicts} 个冲突标注需要人工审核
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* 标注者绩效 */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                标注者绩效分析
              </Typography>
              
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>标注者</TableCell>
                      <TableCell>标注数量</TableCell>
                      <TableCell>平均置信度</TableCell>
                      <TableCell>质量评分</TableCell>
                      <TableCell>一致性评分</TableCell>
                      <TableCell>综合评价</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {qualityReport.annotator_performance.map((annotator: any) => (
                      <TableRow key={annotator.annotator_id} hover>
                        <TableCell>
                          <Typography variant="subtitle2">
                            {annotator.name}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            ID: {annotator.annotator_id}
                          </Typography>
                        </TableCell>
                        <TableCell>{annotator.total_annotations}</TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <LinearProgress
                              variant="determinate"
                              value={annotator.avg_confidence * 100}
                              sx={{ width: 60, mr: 1 }}
                            />
                            <Typography variant="body2">
                              {(annotator.avg_confidence * 100).toFixed(0)}%
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={`${(annotator.quality_score * 100).toFixed(1)}%`}
                            color={getScoreColor(annotator.quality_score) as any}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <LinearProgress
                              variant="determinate"
                              value={annotator.consistency_score * 100}
                              sx={{ width: 60, mr: 1 }}
                            />
                            <Typography variant="body2">
                              {(annotator.consistency_score * 100).toFixed(0)}%
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            {annotator.quality_score >= 0.9 ? (
                              <CheckCircleIcon color="success" sx={{ mr: 1 }} />
                            ) : annotator.quality_score >= 0.8 ? (
                              <WarningIcon color="warning" sx={{ mr: 1 }} />
                            ) : (
                              <WarningIcon color="error" sx={{ mr: 1 }} />
                            )}
                            <Typography variant="body2">
                              {getScoreLabel(annotator.quality_score)}
                            </Typography>
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

        {/* 改进建议 */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                质量改进建议
              </Typography>
              
              <Box>
                {qualityReport.recommendations.map((recommendation: string, index: number) => (
                  <Alert 
                    key={index} 
                    severity="info" 
                    sx={{ mb: 1 }}
                    icon={<AssessmentIcon />}
                  >
                    {recommendation}
                  </Alert>
                ))}
              </Box>

              <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                <Button 
                  variant="contained" 
                  startIcon={<AssessmentIcon />}
                  onClick={() => setReportDialog(true)}
                >
                  生成详细报告
                </Button>
                <Button variant="outlined">
                  导出质量数据
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* 质量报告对话框 */}
      <Dialog open={reportDialog} onClose={() => setReportDialog(false)} maxWidth="lg" fullWidth>
        <DialogTitle>质量分析详细报告</DialogTitle>
        <DialogContent>
          <Typography variant="h6" gutterBottom>
            整体质量评估
          </Typography>
          <Typography variant="body2" paragraph>
            本次分析涵盖了 {qualityReport.annotator_performance.length} 名标注者的工作质量，
            总计 {qualityReport.annotator_performance.reduce((sum: number, p: any) => sum + p.total_annotations, 0)} 条标注记录。
            整体质量评分为 {(qualityReport.overall_score * 100).toFixed(1)}%，属于{getScoreLabel(qualityReport.overall_score)}水平。
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
            主要发现
          </Typography>
          <Typography variant="body2" paragraph>
            • 标注者间一致性达到 {(qualityReport.agreement_metrics.fleiss_kappa * 100).toFixed(1)}%，表现良好
          </Typography>
          <Typography variant="body2" paragraph>
            • 发现 {qualityReport.conflict_analysis.pending_conflicts} 个需要人工审核的冲突标注
          </Typography>
          <Typography variant="body2" paragraph>
            • 建议对质量评分低于85%的标注者进行针对性培训
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setReportDialog(false)}>关闭</Button>
          <Button variant="contained">下载报告</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}