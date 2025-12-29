// 工具权限配置组件

import React, { useState, useEffect } from 'react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Switch } from '../ui/switch';
import { Badge } from '../ui/badge';
import { Alert } from '../ui/alert';
import { Input } from '../ui/input';
import { securityApi } from '../../services/securityApi';

import { logger } from '../../utils/logger'
interface ToolPermission {
  tool_name: string;
  description: string;
  category: string;
  enabled: boolean;
  whitelist_only: boolean;
  requires_approval: boolean;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  allowed_roles: string[];
  usage_count: number;
  last_used?: string;
}

interface ToolWhitelist {
  tool_name: string;
  users: string[];
  roles: string[];
}

export const ToolPermissions: React.FC = () => {
  const [permissions, setPermissions] = useState<ToolPermission[]>([]);
  const [whitelist, setWhitelist] = useState<ToolWhitelist[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  useEffect(() => {
    loadToolPermissions();
  }, []);

  const loadToolPermissions = async () => {
    try {
      setLoading(true);
      const [permsData, whitelistData] = await Promise.all([
        securityApi.getToolPermissions(),
        securityApi.getToolWhitelist()
      ]);
      setPermissions(permsData);
      setWhitelist(whitelistData);
      setError(null);
    } catch (err) {
      setError('加载工具权限失败');
      logger.error('加载工具权限失败:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleEnabled = async (perm: ToolPermission, enabled: boolean) => {
    try {
      await securityApi.updateToolPermission(perm.tool_name, { ...perm, enabled });
      await loadToolPermissions();
    } catch (err) {
      logger.error('更新工具权限失败:', err);
      setError('更新工具权限失败');
    }
  };

  const handleToggleApproval = async (perm: ToolPermission, requiresApproval: boolean) => {
    try {
      await securityApi.updateToolPermission(perm.tool_name, { ...perm, requires_approval: requiresApproval });
      await loadToolPermissions();
    } catch (err) {
      logger.error('更新审批要求失败:', err);
      setError('更新审批要求失败');
    }
  };

  const handleUpdateWhitelist = async (toolName: string, users: string[], roles: string[]) => {
    try {
      await securityApi.updateToolWhitelist(toolName, { users, roles });
      await loadToolPermissions();
    } catch (err) {
      logger.error('更新白名单失败:', err);
      setError('更新白名单失败');
    }
  };

  const getRiskLevelColor = (level: string) => {
    switch (level) {
      case 'critical': return 'text-red-600 bg-red-100';
      case 'high': return 'text-orange-600 bg-orange-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'low': return 'text-green-600 bg-green-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const categories = ['all', ...new Set(permissions.map(p => p.category))];

  const filteredPermissions = permissions.filter(perm => {
    const matchesSearch = perm.tool_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         perm.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || perm.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

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
        <h1 className="text-2xl font-bold">MCP工具权限管理</h1>
        <Button onClick={loadToolPermissions} variant="outline">
          刷新
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <p>{error}</p>
        </Alert>
      )}

      {/* 过滤器 */}
      <Card className="p-4">
        <div className="flex space-x-4">
          <Input
            className="flex-1"
            placeholder="搜索工具..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
          <select
            className="px-3 py-2 border rounded-md"
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
          >
            {categories.map(cat => (
              <option key={cat} value={cat}>
                {cat === 'all' ? '所有分类' : cat}
              </option>
            ))}
          </select>
        </div>
      </Card>

      {/* 工具权限列表 */}
      <div className="space-y-4">
        {filteredPermissions.map((perm) => {
          const wl = whitelist.find(w => w.tool_name === perm.tool_name);
          
          return (
            <Card key={perm.tool_name} className="p-6">
              <div className="space-y-4">
                {/* 工具基本信息 */}
                <div className="flex justify-between items-start">
                  <div>
                    <div className="flex items-center space-x-2">
                      <h3 className="font-semibold text-lg">{perm.tool_name}</h3>
                      <Badge className={getRiskLevelColor(perm.risk_level)}>
                        {perm.risk_level}风险
                      </Badge>
                      <Badge variant="outline">{perm.category}</Badge>
                    </div>
                    <p className="text-gray-600 mt-1">{perm.description}</p>
                    <div className="text-sm text-gray-500 mt-2">
                      使用次数: {perm.usage_count} | 
                      {perm.last_used && ` 最后使用: ${new Date(perm.last_used).toLocaleString()}`}
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2">
                      <span className="text-sm">启用</span>
                      <Switch
                        checked={perm.enabled}
                        onCheckedChange={(checked) => handleToggleEnabled(perm, checked)}
                      />
                    </div>
                  </div>
                </div>

                {/* 安全设置 */}
                <div className="border-t pt-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">需要审批</span>
                      <Switch
                        checked={perm.requires_approval}
                        onCheckedChange={(checked) => handleToggleApproval(perm, checked)}
                        disabled={!perm.enabled}
                      />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-sm">仅白名单</span>
                      <Switch
                        checked={perm.whitelist_only}
                        disabled={!perm.enabled}
                      />
                    </div>
                  </div>
                </div>

                {/* 允许的角色 */}
                <div className="border-t pt-4">
                  <p className="text-sm font-medium mb-2">允许的角色:</p>
                  <div className="flex flex-wrap gap-2">
                    {perm.allowed_roles.map((role) => (
                      <Badge key={role} variant="secondary">
                        {role}
                      </Badge>
                    ))}
                  </div>
                </div>

                {/* 白名单 (如果启用) */}
                {perm.whitelist_only && wl && (
                  <div className="border-t pt-4">
                    <p className="text-sm font-medium mb-2">白名单:</p>
                    <div className="space-y-2">
                      {wl.users.length > 0 && (
                        <div>
                          <span className="text-xs text-gray-500">用户: </span>
                          {wl.users.map((user) => (
                            <Badge key={user} variant="outline" className="ml-1">
                              {user}
                            </Badge>
                          ))}
                        </div>
                      )}
                      {wl.roles.length > 0 && (
                        <div>
                          <span className="text-xs text-gray-500">角色: </span>
                          {wl.roles.map((role) => (
                            <Badge key={role} variant="outline" className="ml-1">
                              {role}
                            </Badge>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
};
