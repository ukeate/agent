import React, { useEffect, useState, useCallback } from 'react'
import { buildApiUrl, apiFetch } from '../utils/apiBase'
import { logger } from '../utils/logger'
import {
  Container,
  Typography,
  Box,
  Paper,
  Button,
  Grid,
  Card,
  CardContent,
  CardActions,
  LinearProgress,
  Alert,
  Tabs,
  Tab,
  Chip,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControlLabel,
  Switch,
} from '@mui/material'
import { documentsService } from '../services/documentsService'
import { message } from 'antd'
import {
  Upload,
  Description,
  CloudUpload,
  Visibility,
  GetApp,
  Delete,
  History,
  Label,
  AccountTree,
  Refresh,
  Close,
  UploadFile,
} from '@mui/icons-material'
import { useDropzone } from 'react-dropzone'

interface DocumentInfo {
  doc_id: string
  title: string
  file_type: string
  file_size?: number
  created_at: string
  status: 'processing' | 'completed' | 'error'
  tags?: string[]
  chunks_count?: number
  processing_info?: {
    chunks?: Array<{
      chunk_id: string
      content: string
      type: string
      index: number
    }>
    auto_tags?: Array<{
      tag: string
      category: string
      confidence: number
    }>
    total_chunks?: number
  }
  version?: {
    version_id: string
    version_number: number
  }
}

interface DocumentProcessingPageProps {}

const DocumentProcessingPage: React.FC<DocumentProcessingPageProps> = () => {
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [uploading, setUploading] = useState(false)
  const [selectedTab, setSelectedTab] = useState(0)
  const [selectedDocument, setSelectedDocument] = useState<DocumentInfo | null>(
    null
  )
  const [dialogOpen, setDialogOpen] = useState(false)
  const [uploadOptions, setUploadOptions] = useState({
    enable_ocr: false,
    extract_images: true,
    auto_tag: true,
    chunk_strategy: 'semantic',
    auto_chunk: true,
    chunk_size: '512',
    chunk_overlap: '50',
    auto_summarize: true,
  })

  const [relationshipsOpen, setRelationshipsOpen] = useState(false)
  const [relationshipsResult, setRelationshipsResult] = useState<any>(null)
  const [versionsOpen, setVersionsOpen] = useState(false)
  const [versionsResult, setVersionsResult] = useState<any>(null)

  const loadDocuments = useCallback(async () => {
    try {
      const response = await apiFetch(buildApiUrl('/api/v1/documents/list'))
      const data = await response.json()
      setDocuments(Array.isArray(data?.documents) ? data.documents : [])
    } catch (error) {
      logger.error('加载文档列表失败:', error)
    }
  }, [])

  useEffect(() => {
    loadDocuments()
  }, [loadDocuments])

  // 文件上传处理
  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      setUploading(true)

      try {
        for (const file of acceptedFiles) {
          try {
            const result = await documentsService.uploadDocument(file, {
              enableOcr: uploadOptions.enable_ocr,
              extractImages: uploadOptions.extract_images,
              autoTag: uploadOptions.auto_tag,
              chunkStrategy: uploadOptions.chunk_strategy as any,
            })

            const newDoc: DocumentInfo = {
              doc_id: result.doc_id,
              title: result.title || file.name,
              file_type:
                result.file_type ||
                result.content_type ||
                file.type ||
                'unknown',
              file_size:
                result.metadata?.file_size || result.metadata?.size_bytes,
              status: 'completed',
              created_at: new Date().toISOString(),
              tags: (result.processing_info?.auto_tags || []).map(
                (t: any) => t.tag
              ),
              processing_info: result.processing_info,
              version: result.version,
            }
            setDocuments(prev => [...prev, newDoc])
          } catch (error) {
            logger.error('上传失败:', error)
          }
        }
      } catch (error) {
        logger.error('上传错误:', error)
      } finally {
        setUploading(false)
      }
    },
    [uploadOptions]
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        ['.docx'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': [
        '.xlsx',
      ],
      'application/vnd.openxmlformats-officedocument.presentationml.presentation':
        ['.pptx'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
      'text/x-python': ['.py'],
      'application/javascript': ['.js'],
      'text/x-java': ['.java'],
    },
  })

  // 查看文档详情
  const handleViewDocument = async (docId: string) => {
    try {
      // 这里可以调用API获取详细信息
      const doc = documents.find(d => d.doc_id === docId)
      if (doc) {
        setSelectedDocument(doc)
        setDialogOpen(true)
      }
    } catch (error) {
      logger.error('获取文档详情失败:', error)
    }
  }

  // 生成标签
  const handleGenerateTags = async (docId: string) => {
    try {
      const current = documents.find(d => d.doc_id === docId)
      const result = await documentsService.generateDocumentTags(
        docId,
        undefined,
        current?.tags || []
      )
      setDocuments(prev =>
        prev.map(doc =>
          doc.doc_id === docId
            ? { ...doc, tags: (result.tags || []).map((t: any) => t.tag) }
            : doc
        )
      )
      message.success('标签生成完成')
    } catch (error) {
      logger.error('生成标签失败:', error)
      message.error('标签生成失败')
    }
  }

  // 分析关系
  const handleAnalyzeRelationships = async (docId: string) => {
    try {
      const result = await documentsService.analyzeDocumentRelationships(docId)
      setRelationshipsResult(result)
      setRelationshipsOpen(true)
    } catch (error) {
      logger.error('关系分析失败:', error)
      message.error('关系分析失败')
    }
  }

  // 获取版本历史
  const handleViewVersions = async (docId: string) => {
    try {
      const result = await documentsService.getDocumentVersionHistory(docId)
      setVersionsResult(result)
      setVersionsOpen(true)
    } catch (error) {
      logger.error('获取版本历史失败:', error)
      message.error('获取版本历史失败')
    }
  }

  // 删除文档
  const handleDeleteDocument = async (docId: string) => {
    try {
      const result = await documentsService.deleteDocument(docId)
      if (result.success) {
        setDocuments(prev => prev.filter(doc => doc.doc_id !== docId))
        message.success(result.message || '文档已删除')
        return
      }
      message.error(result.message || '删除失败')
    } catch (error) {
      logger.error('删除文档失败:', error)
      message.error('删除文档失败')
    }
  }

  const handleBatchUpload = async (files: File[]) => {
    setUploading(true)
    try {
      const result = await documentsService.batchUploadDocuments(files)
      const newDocs: DocumentInfo[] = (result.results || []).map(
        (doc: any) => ({
          doc_id: doc.doc_id,
          title: doc.title || doc.doc_id,
          file_type: doc.file_type || 'unknown',
          status: 'completed' as const,
          created_at: new Date().toISOString(),
        })
      )
      setDocuments(prev => [...newDocs, ...prev])
      message.success(
        `批量上传完成: ${result.success || newDocs.length} 个文件`
      )
    } catch (error) {
      logger.error('批量上传失败:', error)
      message.error('批量上传失败')
    } finally {
      setUploading(false)
    }
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        智能文档处理系统
      </Typography>

      {/* 文件上传区域 */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box
          data-testid="upload-area"
          {...getRootProps()}
          sx={{
            border: '2px dashed #ccc',
            borderRadius: 2,
            p: 4,
            textAlign: 'center',
            cursor: 'pointer',
            backgroundColor: isDragActive ? '#f5f5f5' : 'transparent',
            '&:hover': { backgroundColor: '#f9f9f9' },
          }}
        >
          <input {...getInputProps()} />
          <CloudUpload sx={{ fontSize: 48, color: '#666', mb: 2 }} />
          {uploading ? (
            <Box>
              <Typography>正在上传和处理...</Typography>
              <LinearProgress sx={{ mt: 2 }} />
            </Box>
          ) : isDragActive ? (
            <Typography>将文件拖放到这里...</Typography>
          ) : (
            <Box>
              <Typography variant="h6">拖拽文件到这里或点击选择</Typography>
              <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                支持 PDF, Word, Excel, PowerPoint, 文本, Markdown, 代码文件
              </Typography>
            </Box>
          )}
        </Box>

        {/* 上传选项 */}
        <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <FormControlLabel
            control={
              <Switch
                checked={uploadOptions.enable_ocr}
                onChange={e =>
                  setUploadOptions(prev => ({
                    ...prev,
                    enable_ocr: e.target.checked,
                  }))
                }
              />
            }
            label="启用OCR"
          />
          <FormControlLabel
            control={
              <Switch
                checked={uploadOptions.extract_images}
                onChange={e =>
                  setUploadOptions(prev => ({
                    ...prev,
                    extract_images: e.target.checked,
                  }))
                }
              />
            }
            label="提取图像"
          />
          <FormControlLabel
            control={
              <Switch
                checked={uploadOptions.auto_tag}
                onChange={e =>
                  setUploadOptions(prev => ({
                    ...prev,
                    auto_tag: e.target.checked,
                  }))
                }
              />
            }
            label="自动标签"
          />
        </Box>

        {/* 新增：批量上传按钮 */}
        <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={<UploadFile />}
            onClick={() => {
              const input = document.createElement('input')
              input.type = 'file'
              input.multiple = true
              input.accept = '.pdf,.docx,.xlsx,.pptx,.txt,.md,.py,.js,.java'
              input.onchange = e => {
                const files = Array.from(
                  (e.target as HTMLInputElement).files || []
                )
                if (files.length > 0) {
                  handleBatchUpload(files)
                }
              }
              input.click()
            }}
            disabled={uploading}
          >
            批量上传文档
          </Button>
        </Box>
      </Paper>

      {/* 标签栏 */}
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={selectedTab}
          onChange={(_, newValue) => setSelectedTab(newValue)}
          indicatorColor="primary"
          textColor="primary"
        >
          <Tab label="所有文档" />
          <Tab label="最近上传" />
          <Tab label="PDF文档" />
          <Tab label="代码文件" />
        </Tabs>
      </Paper>

      {/* 文档列表 */}
      <Grid container spacing={3}>
        {documents.map(doc => (
          <Grid item xs={12} md={6} lg={4} key={doc.doc_id}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 2 }}>
                  <Description sx={{ mr: 1, color: 'primary.main' }} />
                  <Box sx={{ flexGrow: 1 }}>
                    <Typography variant="h6" noWrap>
                      {doc.title}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      {doc.file_type.toUpperCase()} •{' '}
                      {new Date(doc.created_at).toLocaleDateString()}
                    </Typography>
                  </Box>
                  <Chip
                    label={doc.status}
                    color={doc.status === 'completed' ? 'success' : 'warning'}
                    size="small"
                  />
                </Box>

                {doc.tags && (
                  <Box sx={{ mb: 2 }}>
                    {doc.tags.slice(0, 3).map(tag => (
                      <Chip
                        key={tag}
                        label={tag}
                        size="small"
                        variant="outlined"
                        sx={{ mr: 0.5, mb: 0.5 }}
                      />
                    ))}
                    {doc.tags.length > 3 && (
                      <Chip
                        label={`+${doc.tags.length - 3}`}
                        size="small"
                        variant="outlined"
                      />
                    )}
                  </Box>
                )}

                {doc.processing_info && (
                  <Typography variant="body2" color="textSecondary">
                    分块数量: {doc.processing_info.total_chunks || 0}
                  </Typography>
                )}
              </CardContent>

              <CardActions>
                <Button
                  size="small"
                  startIcon={<Visibility />}
                  onClick={() => handleViewDocument(doc.doc_id)}
                >
                  查看
                </Button>
                <IconButton
                  size="small"
                  onClick={() => handleGenerateTags(doc.doc_id)}
                  title="生成标签"
                >
                  <Label />
                </IconButton>
                <IconButton
                  size="small"
                  onClick={() => handleAnalyzeRelationships(doc.doc_id)}
                  title="分析关系"
                >
                  <AccountTree />
                </IconButton>
                <IconButton
                  size="small"
                  onClick={() => handleViewVersions(doc.doc_id)}
                  title="版本历史"
                >
                  <History />
                </IconButton>
                <IconButton
                  size="small"
                  onClick={() => handleDeleteDocument(doc.doc_id)}
                  title="删除"
                  color="error"
                >
                  <Delete />
                </IconButton>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      {documents.length === 0 && (
        <Paper sx={{ p: 6, textAlign: 'center', mt: 3 }}>
          <Description sx={{ fontSize: 64, color: '#ccc', mb: 2 }} />
          <Typography variant="h6" color="textSecondary">
            还没有文档
          </Typography>
          <Typography variant="body2" color="textSecondary">
            拖拽文件到上方区域开始上传和处理
          </Typography>
        </Paper>
      )}

      {/* 文档详情对话框 */}
      <Dialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          文档详情
          <IconButton
            sx={{ position: 'absolute', right: 8, top: 8 }}
            onClick={() => setDialogOpen(false)}
          >
            <Close />
          </IconButton>
        </DialogTitle>
        <DialogContent>
          {selectedDocument && (
            <Box>
              <Typography variant="h6" gutterBottom>
                {selectedDocument.title}
              </Typography>

              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">
                    文件类型: {selectedDocument.file_type}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">
                    状态: {selectedDocument.status}
                  </Typography>
                </Grid>
              </Grid>

              {selectedDocument.processing_info && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    处理信息
                  </Typography>

                  {selectedDocument.processing_info.auto_tags && (
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" gutterBottom>
                        自动标签:
                      </Typography>
                      {selectedDocument.processing_info.auto_tags.map(tag => (
                        <Chip
                          key={tag.tag}
                          label={`${tag.tag} (${(tag.confidence * 100).toFixed(1)}%)`}
                          size="small"
                          sx={{ mr: 1, mb: 1 }}
                        />
                      ))}
                    </Box>
                  )}

                  {selectedDocument.processing_info.chunks && (
                    <Box>
                      <Typography variant="body2" gutterBottom>
                        内容分块 (
                        {selectedDocument.processing_info.total_chunks}):
                      </Typography>
                      <List dense>
                        {selectedDocument.processing_info.chunks
                          .slice(0, 5)
                          .map(chunk => (
                            <ListItem key={chunk.chunk_id}>
                              <ListItemText
                                primary={`分块 ${chunk.index + 1}`}
                                secondary={chunk.content}
                                secondaryTypographyProps={{
                                  noWrap: true,
                                  style: { maxWidth: 400 },
                                }}
                              />
                            </ListItem>
                          ))}
                      </List>
                    </Box>
                  )}
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>关闭</Button>
        </DialogActions>
      </Dialog>

      <Dialog
        open={relationshipsOpen}
        onClose={() => setRelationshipsOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          关系分析结果
          <IconButton
            sx={{ position: 'absolute', right: 8, top: 8 }}
            onClick={() => setRelationshipsOpen(false)}
          >
            <Close />
          </IconButton>
        </DialogTitle>
        <DialogContent>
          {relationshipsResult ? (
            <Box>
              <Typography variant="body2" color="textSecondary">
                关系数: {(relationshipsResult.relationships || []).length} •
                聚类数: {(relationshipsResult.clusters || []).length}
              </Typography>
            </Box>
          ) : null}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRelationshipsOpen(false)}>关闭</Button>
        </DialogActions>
      </Dialog>

      <Dialog
        open={versionsOpen}
        onClose={() => setVersionsOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          版本历史
          <IconButton
            sx={{ position: 'absolute', right: 8, top: 8 }}
            onClick={() => setVersionsOpen(false)}
          >
            <Close />
          </IconButton>
        </DialogTitle>
        <DialogContent>
          {versionsResult?.versions ? (
            <List dense>
              {versionsResult.versions.map((v: any) => (
                <ListItem key={v.version_id}>
                  <ListItemText
                    primary={`v${v.version_number}${v.is_current ? '（当前）' : ''}`}
                    secondary={v.change_summary || v.created_at}
                  />
                </ListItem>
              ))}
            </List>
          ) : null}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setVersionsOpen(false)}>关闭</Button>
        </DialogActions>
      </Dialog>
    </Container>
  )
}

export default DocumentProcessingPage
