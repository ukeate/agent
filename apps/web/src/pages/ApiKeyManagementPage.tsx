import React, { useState, useEffect } from 'react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Input } from '../components/ui/input'
import { Label } from '../components/ui/label'
import { Textarea } from '../components/ui/textarea'
import { Badge } from '../components/ui/badge'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '../components/ui/dialog'
import { Plus, Eye, EyeOff, Copy, Trash2, Key, Shield, Clock } from 'lucide-react'
import { toast } from 'react-hot-toast'

interface APIKey {
  id: string
  name: string
  key: string
  created_at: string
  expires_at: string | null
  permissions: string[]
  status: 'active' | 'expired' | 'revoked'
}

interface CreateAPIKeyRequest {
  name: string
  description?: string
  expires_in_days?: number
  permissions: string[]
}

const ApiKeyManagementPage: React.FC = () => {
  const [apiKeys, setApiKeys] = useState<APIKey[]>([])
  const [loading, setLoading] = useState(false)
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false)
  const [visibleKeys, setVisibleKeys] = useState<Set<string>>(new Set())
  const [newKey, setNewKey] = useState<CreateAPIKeyRequest>({
    name: '',
    description: '',
    expires_in_days: 30,
    permissions: []
  })

  const availablePermissions = [
    'system:read',
    'system:write',
    'system:admin',
    'memory:read',
    'memory:write',
    'rag:search',
    'agent:execute',
    'experiment:read',
    'experiment:write'
  ]

  // 模拟数据初始化
  useEffect(() => {
    loadApiKeys()
  }, [])

  const loadApiKeys = async () => {
    setLoading(true)
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 800))
      
      // 模拟数据
      const mockKeys: APIKey[] = [
        {
          id: '1',
          name: 'Production API Key',
          key: 'sk_prod_1a2b3c4d5e6f7g8h9i0j',
          created_at: '2025-08-20T10:00:00Z',
          expires_at: '2025-11-20T10:00:00Z',
          permissions: ['system:read', 'memory:read', 'rag:search'],
          status: 'active'
        },
        {
          id: '2',
          name: 'Development Key',
          key: 'sk_dev_a1b2c3d4e5f6g7h8i9j0',
          created_at: '2025-08-15T14:30:00Z',
          expires_at: null,
          permissions: ['system:admin'],
          status: 'active'
        },
        {
          id: '3',
          name: 'Legacy Key',
          key: 'sk_legacy_z9y8x7w6v5u4t3s2r1q0',
          created_at: '2025-07-01T09:00:00Z',
          expires_at: '2025-08-01T09:00:00Z',
          permissions: ['system:read'],
          status: 'expired'
        }
      ]
      
      setApiKeys(mockKeys)
    } catch (error) {
      console.error('Failed to load API keys:', error)
      toast.error('加载API密钥失败')
    } finally {
      setLoading(false)
    }
  }

  const createApiKey = async () => {
    if (!newKey.name.trim()) {
      toast.error('请输入API密钥名称')
      return
    }

    setLoading(true)
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      const createdKey: APIKey = {
        id: Date.now().toString(),
        name: newKey.name,
        key: `sk_${Math.random().toString(36).substring(2, 15)}${Math.random().toString(36).substring(2, 15)}`,
        created_at: new Date().toISOString(),
        expires_at: newKey.expires_in_days 
          ? new Date(Date.now() + newKey.expires_in_days * 24 * 60 * 60 * 1000).toISOString()
          : null,
        permissions: newKey.permissions,
        status: 'active'
      }

      setApiKeys(prev => [...prev, createdKey])
      setIsCreateDialogOpen(false)
      setNewKey({
        name: '',
        description: '',
        expires_in_days: 30,
        permissions: []
      })
      toast.success('API密钥创建成功')
    } catch (error) {
      console.error('Failed to create API key:', error)
      toast.error('创建API密钥失败')
    } finally {
      setLoading(false)
    }
  }

  const revokeApiKey = async (keyId: string) => {
    if (!confirm('确定要撤销这个API密钥吗？此操作不可撤销。')) {
      return
    }

    setLoading(true)
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 500))
      
      setApiKeys(prev => 
        prev.map(key => 
          key.id === keyId 
            ? { ...key, status: 'revoked' as const }
            : key
        )
      )
      toast.success('API密钥已撤销')
    } catch (error) {
      console.error('Failed to revoke API key:', error)
      toast.error('撤销API密钥失败')
    } finally {
      setLoading(false)
    }
  }

  const toggleKeyVisibility = (keyId: string) => {
    setVisibleKeys(prev => {
      const newSet = new Set(prev)
      if (newSet.has(keyId)) {
        newSet.delete(keyId)
      } else {
        newSet.add(keyId)
      }
      return newSet
    })
  }

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      toast.success('已复制到剪贴板')
    } catch (error) {
      toast.error('复制失败')
    }
  }

  const maskApiKey = (key: string) => {
    return `${key.substring(0, 8)}...${key.substring(key.length - 4)}`
  }

  const getStatusBadge = (status: string) => {
    const statusConfig = {
      active: { variant: 'default' as const, text: '活跃' },
      expired: { variant: 'secondary' as const, text: '已过期' },
      revoked: { variant: 'destructive' as const, text: '已撤销' }
    }
    
    const config = statusConfig[status as keyof typeof statusConfig]
    return <Badge variant={config.variant}>{config.text}</Badge>
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('zh-CN', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const togglePermission = (permission: string) => {
    setNewKey(prev => ({
      ...prev,
      permissions: prev.permissions.includes(permission)
        ? prev.permissions.filter(p => p !== permission)
        : [...prev.permissions, permission]
    }))
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">API密钥管理</h1>
          <p className="text-gray-600 mt-2">
            管理您的API密钥，控制访问权限和安全性
          </p>
        </div>

        <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              创建API密钥
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-md">
            <DialogHeader>
              <DialogTitle>创建新的API密钥</DialogTitle>
              <DialogDescription>
                创建一个新的API密钥来访问我们的服务
              </DialogDescription>
            </DialogHeader>
            
            <div className="space-y-4">
              <div>
                <Label htmlFor="name">密钥名称 *</Label>
                <Input
                  id="name"
                  placeholder="例如: Production API Key"
                  value={newKey.name}
                  onChange={(e) => setNewKey(prev => ({ ...prev, name: e.target.value }))}
                />
              </div>

              <div>
                <Label htmlFor="description">描述（可选）</Label>
                <Textarea
                  id="description"
                  placeholder="描述此API密钥的用途"
                  value={newKey.description}
                  onChange={(e) => setNewKey(prev => ({ ...prev, description: e.target.value }))}
                />
              </div>

              <div>
                <Label htmlFor="expires">过期时间（天）</Label>
                <Input
                  id="expires"
                  type="number"
                  min="1"
                  max="365"
                  value={newKey.expires_in_days}
                  onChange={(e) => setNewKey(prev => ({ ...prev, expires_in_days: parseInt(e.target.value) || 30 }))}
                />
              </div>

              <div>
                <Label>权限</Label>
                <div className="grid grid-cols-2 gap-2 mt-2">
                  {availablePermissions.map((permission) => (
                    <label key={permission} className="flex items-center space-x-2 text-sm">
                      <input
                        type="checkbox"
                        checked={newKey.permissions.includes(permission)}
                        onChange={() => togglePermission(permission)}
                        className="rounded"
                      />
                      <span>{permission}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="flex justify-end space-x-2 pt-4">
                <Button 
                  variant="outline" 
                  onClick={() => setIsCreateDialogOpen(false)}
                >
                  取消
                </Button>
                <Button 
                  onClick={createApiKey}
                  disabled={loading || !newKey.name.trim()}
                >
                  {loading ? '创建中...' : '创建密钥'}
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      {/* 统计卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">总密钥数</p>
                <p className="text-2xl font-bold">{apiKeys.length}</p>
              </div>
              <Key className="h-8 w-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">活跃密钥</p>
                <p className="text-2xl font-bold text-green-600">
                  {apiKeys.filter(key => key.status === 'active').length}
                </p>
              </div>
              <Shield className="h-8 w-8 text-green-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">即将过期</p>
                <p className="text-2xl font-bold text-yellow-600">
                  {apiKeys.filter(key => {
                    if (!key.expires_at) return false
                    const expiresAt = new Date(key.expires_at)
                    const now = new Date()
                    const daysToExpiry = (expiresAt.getTime() - now.getTime()) / (1000 * 60 * 60 * 24)
                    return daysToExpiry <= 7 && daysToExpiry > 0
                  }).length}
                </p>
              </div>
              <Clock className="h-8 w-8 text-yellow-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">已撤销</p>
                <p className="text-2xl font-bold text-red-600">
                  {apiKeys.filter(key => key.status === 'revoked').length}
                </p>
              </div>
              <Trash2 className="h-8 w-8 text-red-600" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* API密钥列表 */}
      <Card>
        <CardHeader>
          <CardTitle>API密钥列表</CardTitle>
          <CardDescription>
            管理您的所有API密钥
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading && apiKeys.length === 0 ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
              <p className="text-gray-600 mt-2">加载中...</p>
            </div>
          ) : apiKeys.length === 0 ? (
            <div className="text-center py-8">
              <Key className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600">暂无API密钥</p>
              <p className="text-sm text-gray-500">创建您的第一个API密钥来开始使用</p>
            </div>
          ) : (
            <div className="space-y-4">
              {apiKeys.map((apiKey) => (
                <div
                  key={apiKey.id}
                  className="border rounded-lg p-4 space-y-3"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <h3 className="font-semibold">{apiKey.name}</h3>
                      {getStatusBadge(apiKey.status)}
                    </div>
                    <div className="flex items-center space-x-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => toggleKeyVisibility(apiKey.id)}
                      >
                        {visibleKeys.has(apiKey.id) ? (
                          <EyeOff className="h-4 w-4" />
                        ) : (
                          <Eye className="h-4 w-4" />
                        )}
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => copyToClipboard(apiKey.key)}
                      >
                        <Copy className="h-4 w-4" />
                      </Button>
                      {apiKey.status === 'active' && (
                        <Button
                          variant="destructive"
                          size="sm"
                          onClick={() => revokeApiKey(apiKey.id)}
                          disabled={loading}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      )}
                    </div>
                  </div>

                  <div className="font-mono text-sm bg-gray-100 p-2 rounded">
                    {visibleKeys.has(apiKey.id) ? apiKey.key : maskApiKey(apiKey.key)}
                  </div>

                  <div className="grid grid-cols-2 gap-4 text-sm text-gray-600">
                    <div>
                      <p><strong>创建时间:</strong> {formatDate(apiKey.created_at)}</p>
                      {apiKey.expires_at && (
                        <p><strong>过期时间:</strong> {formatDate(apiKey.expires_at)}</p>
                      )}
                    </div>
                    <div>
                      <p><strong>权限:</strong></p>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {apiKey.permissions.map((permission) => (
                          <Badge key={permission} variant="outline" className="text-xs">
                            {permission}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

export default ApiKeyManagementPage