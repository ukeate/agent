import apiClient from './apiClient';

// ==================== 类型定义 ====================

export interface FileUploadData {
  file_id: string;
  filename: string;
  file_size: number;
  content_type: string;
  mime_type: string;
  upload_time: string;
  openai_file_id?: string;
}

export interface FileUploadResponse {
  success: boolean;
  message: string;
  data: FileUploadData;
}

export interface BatchUploadResult {
  file_id: string;
  filename: string;
  file_size: number;
  content_type: string;
  status: string;
}

export interface BatchUploadError {
  filename: string;
  error: string;
}

export interface BatchUploadResponse {
  success: boolean;
  uploaded_count: number;
  error_count: number;
  results: BatchUploadResult[];
  errors: BatchUploadError[];
}

export interface FileListItem {
  file_id: string;
  filename: string;
  file_size: number;
  created_at: number;
  modified_at: number;
  file_path: string;
}

export interface FileListResponse {
  success: boolean;
  data: {
    files: FileListItem[];
    total: number;
    limit: number;
    offset: number;
  };
}

export interface FileInfoResponse {
  success: boolean;
  data: any;
}

export interface DeleteFileResponse {
  success: boolean;
  message: string;
}

export interface CleanupResponse {
  success: boolean;
  message: string;
  deleted_count: number;
}

export interface FileStats {
  total_files: number;
  total_size: number;
  total_size_mb: number;
  file_types: Record<string, number>;
  upload_path: string;
}

export interface FileStatsResponse {
  success: boolean;
  data: FileStats;
}

// ==================== Service Class ====================

class FilesService {
  private baseUrl = '/files';

  // 单个文件上传
  async uploadFile(file: File): Promise<FileUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post(`${this.baseUrl}/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  }

  // 批量文件上传
  async uploadMultipleFiles(files: File[]): Promise<BatchUploadResponse> {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    const response = await apiClient.post(`${this.baseUrl}/upload/batch`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  }

  // 获取文件列表
  async listFiles(params?: {
    limit?: number;
    offset?: number;
  }): Promise<FileListResponse> {
    const response = await apiClient.get(`${this.baseUrl}/list`, { params });
    return response.data;
  }

  // 获取文件信息
  async getFileInfo(fileId: string): Promise<FileInfoResponse> {
    const response = await apiClient.get(`${this.baseUrl}/${fileId}`);
    return response.data;
  }

  // 下载文件
  async downloadFile(fileId: string): Promise<Blob> {
    const response = await apiClient.get(`${this.baseUrl}/${fileId}/download`, {
      responseType: 'blob'
    });
    return response.data;
  }

  // 删除文件
  async deleteFile(fileId: string): Promise<DeleteFileResponse> {
    const response = await apiClient.delete(`${this.baseUrl}/${fileId}`);
    return response.data;
  }

  // 清理旧文件
  async cleanupOldFiles(days: number = 7): Promise<CleanupResponse> {
    const response = await apiClient.post(`${this.baseUrl}/cleanup`, null, {
      params: { days }
    });
    return response.data;
  }

  // 获取文件统计
  async getFileStats(): Promise<FileStatsResponse> {
    const response = await apiClient.get(`${this.baseUrl}/stats/summary`);
    return response.data;
  }
}

// ==================== 导出 ====================

export const filesService = new FilesService();
export default filesService;