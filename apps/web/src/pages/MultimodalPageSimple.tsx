import React from 'react';
import { Card, Typography } from 'antd';

const { Title } = Typography;

const MultimodalPageSimple: React.FC = () => {
  return (
    <div className="p-6">
      <Title level={2}>GPT-4o 多模态API集成</Title>
      <Card>
        <p>多模态页面测试版本 - 基础功能正常</p>
        <p>支持图像、文档、视频等多种内容处理</p>
      </Card>
    </div>
  );
};

export default MultimodalPageSimple;