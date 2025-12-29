import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { AxiosResponse } from 'axios';

const apiClientMocks = vi.hoisted(() => ({
  postMock: vi.fn(),
  getMock: vi.fn(),
  deleteMock: vi.fn()
}));

vi.mock('../../services/apiClient', () => ({
  __esModule: true,
  default: {
    post: apiClientMocks.postMock,
    get: apiClientMocks.getMock,
    delete: apiClientMocks.deleteMock
  }
}));

import { filesService } from '../../services/filesService';

interface MockResponse<T> extends AxiosResponse<T> {
  data: T;
}

const { postMock, getMock, deleteMock } = apiClientMocks;

describe('filesService', () => {
  beforeEach(() => {
    postMock.mockReset();
    getMock.mockReset();
    deleteMock.mockReset();
  });

  it('调用 uploadFile 时正确发送请求', async () => {
    const mockData = { success: true };
    postMock.mockResolvedValue({ data: mockData } as MockResponse<typeof mockData>);

    const file = new File(['hello'], 'test.txt', { type: 'text/plain' });
    const result = await filesService.uploadFile(file);

    expect(postMock).toHaveBeenCalledWith(
      '/api/v1/files/upload',
      expect.any(FormData),
      expect.objectContaining({ headers: { 'Content-Type': 'multipart/form-data' } })
    );
    expect(result).toEqual(mockData);
  });

  it('调用 uploadMultipleFiles 时正确序列化文件', async () => {
    const mockData = { success: true, uploaded_count: 2, error_count: 0, results: [], errors: [] };
    postMock.mockResolvedValue({ data: mockData } as MockResponse<typeof mockData>);

    const files = [
      new File(['foo'], 'a.txt', { type: 'text/plain' }),
      new File(['bar'], 'b.txt', { type: 'text/plain' })
    ];

    const result = await filesService.uploadMultipleFiles(files);

    expect(postMock).toHaveBeenCalledWith(
      '/api/v1/files/upload/batch',
      expect.any(FormData),
      expect.objectContaining({ headers: { 'Content-Type': 'multipart/form-data' } })
    );

    const formDataArg = postMock.mock.calls[0][1] as FormData;
    const appendedFiles = formDataArg.getAll('files');
    expect(appendedFiles).toHaveLength(2);
    expect(result).toEqual(mockData);
  });

  it('调用 listFiles 时传递查询参数', async () => {
    const mockData = { success: true, data: { files: [], total: 0, limit: 10, offset: 0 } };
    getMock.mockResolvedValue({ data: mockData } as MockResponse<typeof mockData>);

    const params = { limit: 10, offset: 5 };
    const result = await filesService.listFiles(params);

    expect(getMock).toHaveBeenCalledWith('/api/v1/files/list', { params });
    expect(result).toEqual(mockData);
  });

  it('调用 getFileInfo 时访问正确路径', async () => {
    const mockData = { success: true, data: {} };
    getMock.mockResolvedValue({ data: mockData } as MockResponse<typeof mockData>);

    const result = await filesService.getFileInfo('abc');

    expect(getMock).toHaveBeenCalledWith('/api/v1/files/abc');
    expect(result).toEqual(mockData);
  });

  it('调用 downloadFile 时设置 responseType', async () => {
    const mockBlob = new Blob(['content']);
    getMock.mockResolvedValue({ data: mockBlob } as MockResponse<Blob>);

    const result = await filesService.downloadFile('file-id');

    expect(getMock).toHaveBeenCalledWith('/api/v1/files/file-id/download', {
      responseType: 'blob'
    });
    expect(result).toBe(mockBlob);
  });

  it('调用 deleteFile 时使用 DELETE 方法', async () => {
    const mockData = { success: true };
    deleteMock.mockResolvedValue({ data: mockData } as MockResponse<typeof mockData>);

    const result = await filesService.deleteFile('file-id');

    expect(deleteMock).toHaveBeenCalledWith('/api/v1/files/file-id');
    expect(result).toEqual(mockData);
  });

  it('调用 cleanupOldFiles 时传递 days 参数', async () => {
    const mockData = { success: true, message: 'ok', deleted_count: 1 };
    postMock.mockResolvedValue({ data: mockData } as MockResponse<typeof mockData>);

    const result = await filesService.cleanupOldFiles(3);

    expect(postMock).toHaveBeenCalledWith('/api/v1/files/cleanup', null, {
      params: { days: 3 }
    });
    expect(result).toEqual(mockData);
  });

  it('调用 getFileStats 时访问统计端点', async () => {
    const mockData = { success: true, data: { total_files: 0, total_size: 0, total_size_mb: 0, file_types: {}, upload_path: '' } };
    getMock.mockResolvedValue({ data: mockData } as MockResponse<typeof mockData>);

    const result = await filesService.getFileStats();

    expect(getMock).toHaveBeenCalledWith('/api/v1/files/stats/summary');
    expect(result).toEqual(mockData);
  });
});
