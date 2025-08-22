// 安全管理页面

import React, { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/Tabs';
import { SecurityDashboard } from '../components/security/SecurityDashboard';
import { APIKeyManager } from '../components/security/APIKeyManager';
import { ToolPermissions } from '../components/security/ToolPermissions';
import { SecurityAlerts } from '../components/security/SecurityAlerts';

const SecurityPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('dashboard');

  return (
    <div className="container mx-auto p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">安全管理中心</h1>
        <p className="text-gray-600 mt-2">
          管理API安全、工具权限、安全告警和访问控制
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="dashboard">安全概览</TabsTrigger>
          <TabsTrigger value="api-keys">API密钥</TabsTrigger>
          <TabsTrigger value="tool-permissions">工具权限</TabsTrigger>
          <TabsTrigger value="alerts">安全告警</TabsTrigger>
        </TabsList>

        <TabsContent value="dashboard" className="mt-6">
          <SecurityDashboard />
        </TabsContent>

        <TabsContent value="api-keys" className="mt-6">
          <APIKeyManager />
        </TabsContent>

        <TabsContent value="tool-permissions" className="mt-6">
          <ToolPermissions />
        </TabsContent>

        <TabsContent value="alerts" className="mt-6">
          <SecurityAlerts />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default SecurityPage;