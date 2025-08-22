import React from 'react';
import { Card, Typography } from 'antd';

const { Title, Text } = Typography;

const VectorAdvancedTestPage: React.FC = () => {
  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>向量高级功能测试页面</Title>
      <Card>
        <Text>如果你看到这个页面，说明路由工作正常。</Text>
      </Card>
    </div>
  );
};

export default VectorAdvancedTestPage;