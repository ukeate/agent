// API密钥管理组件

import React, { useState, useEffect } from 'react';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import { Alert } from '../ui/Alert';
import { Badge } from '../ui/Badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../ui/Dialog';
import { securityApi } from '../../services/securityApi';

interface APIKey {
  id: string;
  name: string;
  key: string;
  created_at: string;
  last_used_at?: string;
  expires_at?: string;
  permissions: string[];
  rate_limits: {
    requests_per_minute: number;
    requests_per_hour: number;
  };
  status: 'active' | 'expired' | 'revoked';
}

export const APIKeyManager: React.FC = () => {
  const [apiKeys, setApiKeys] = useState<APIKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [newKeyData, setNewKeyData] = useState({
    name: '',
    permissions: [] as string[],
    expires_in_days: 30
  });
  const [createdKey, setCreatedKey] = useState<string | null>(null);

  useEffect(() => {
    loadAPIKeys();
  }, []);

  const loadAPIKeys = async () => {
    try {
      setLoading(true);
      const keys = await securityApi.getAPIKeys();
      setApiKeys(keys);
      setError(null);
    } catch (err) {
      setError('加载API密钥失败');
      console.error('Error loading API keys:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateKey = async () => {
    try {
      const result = await securityApi.createAPIKey(newKeyData);
      setCreatedKey(result.key);
      setNewKeyData({ name: '', permissions: [], expires_in_days: 30 });
      await loadAPIKeys();
    } catch (err) {
      console.error('Error creating API key:', err);
      setError('创建API密钥失败');
    }
  };

  const handleRevokeKey = async (keyId: string) => {
    if (!confirm('确定要撤销这个API密钥吗？此操作不可恢复。')) {
      return;
    }
    
    try {
      await securityApi.revokeAPIKey(keyId);
      await loadAPIKeys();
    } catch (err) {
      console.error('Error revoking API key:', err);
      setError('撤销API密钥失败');
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    // 可以添加一个提示
  };

  const availablePermissions = [
    'read:agents',
    'write:agents',
    'read:workflows',
    'write:workflows',
    'read:rag',
    'write:rag',
    'execute:tools',
    'admin:security'
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">API密钥管理</h1>
        <Button onClick={() => setShowCreateDialog(true)}>
          创建新密钥
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <p>{error}</p>
        </Alert>
      )}

      {createdKey && (
        <Alert>
          <div>
            <p className="font-semibold mb-2">新API密钥已创建！请立即复制保存：</p>
            <div className="flex items-center space-x-2">
              <code className="bg-gray-100 px-2 py-1 rounded text-sm">{createdKey}</code>
              <Button
                size="sm"
                variant="outline"
                onClick={() => copyToClipboard(createdKey)}
              >
                复制
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setCreatedKey(null)}
              >
                关闭
              </Button>
            </div>
            <p className="text-sm text-gray-500 mt-2">
              注意：此密钥只会显示一次，请妥善保管
            </p>
          </div>
        </Alert>
      )}

      <div className="grid gap-4">
        {apiKeys.length === 0 ? (
          <Card className="p-6 text-center text-gray-500">
            暂无API密钥
          </Card>
        ) : (
          apiKeys.map((key) => (
            <Card key={key.id} className="p-6">
              <div className="flex justify-between items-start">
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <h3 className="font-semibold">{key.name}</h3>
                    <Badge variant={
                      key.status === 'active' ? 'default' :
                      key.status === 'expired' ? 'secondary' : 'destructive'
                    }>
                      {key.status}
                    </Badge>
                  </div>
                  
                  <div className="text-sm text-gray-500">
                    <p>创建时间: {new Date(key.created_at).toLocaleString()}</p>
                    {key.last_used_at && (
                      <p>最后使用: {new Date(key.last_used_at).toLocaleString()}</p>
                    )}
                    {key.expires_at && (
                      <p>过期时间: {new Date(key.expires_at).toLocaleString()}</p>
                    )}
                  </div>
                  
                  <div className="text-sm">
                    <p className="font-medium">速率限制:</p>
                    <p className="text-gray-500">
                      {key.rate_limits.requests_per_minute} 请求/分钟 | 
                      {key.rate_limits.requests_per_hour} 请求/小时
                    </p>
                  </div>
                  
                  <div className="text-sm">
                    <p className="font-medium">权限:</p>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {key.permissions.map((perm) => (
                        <Badge key={perm} variant="outline" className="text-xs">
                          {perm}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
                
                <div className="flex space-x-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => copyToClipboard(key.key)}
                  >
                    复制密钥
                  </Button>
                  {key.status === 'active' && (
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={() => handleRevokeKey(key.id)}
                    >
                      撤销
                    </Button>
                  )}
                </div>
              </div>
            </Card>
          ))
        )}
      </div>

      {/* 创建API密钥对话框 */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>创建新API密钥</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">名称</label>
              <Input
                value={newKeyData.name}
                onChange={(e) => setNewKeyData({ ...newKeyData, name: e.target.value })}
                placeholder="例如: 生产环境密钥"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">有效期（天）</label>
              <Input
                type="number"
                value={newKeyData.expires_in_days}
                onChange={(e) => setNewKeyData({
                  ...newKeyData,
                  expires_in_days: parseInt(e.target.value)
                })}
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">权限</label>
              <div className="space-y-2">
                {availablePermissions.map((perm) => (
                  <label key={perm} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={newKeyData.permissions.includes(perm)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setNewKeyData({
                            ...newKeyData,
                            permissions: [...newKeyData.permissions, perm]
                          });
                        } else {
                          setNewKeyData({
                            ...newKeyData,
                            permissions: newKeyData.permissions.filter(p => p !== perm)
                          });
                        }
                      }}
                    />
                    <span className="text-sm">{perm}</span>
                  </label>
                ))}
              </div>
            </div>
            
            <div className="flex justify-end space-x-2">
              <Button
                variant="outline"
                onClick={() => setShowCreateDialog(false)}
              >
                取消
              </Button>
              <Button onClick={handleCreateKey}>
                创建
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};