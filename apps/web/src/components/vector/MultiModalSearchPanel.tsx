/**
 * 多模态搜索面板
 * 展示图像、音频、文本的跨模态搜索功能
 */

import React, { useState } from 'react';
import { Card, Upload, Button, Input, Space, Alert, Tabs, message } from 'antd';
import { PictureOutlined, AudioOutlined, FileTextOutlined } from '@ant-design/icons';

const MultiModalSearchPanel: React.FC = () => {
  const [searchMode, setSearchMode] = useState('text-to-image');

  return (
    <div>
      <Alert
        message="多模态搜索功能"
        description="支持文本、图像、音频之间的跨模态搜索，实现图文互搜、语音检索等功能。"
        variant="default"
        showIcon
        style={{ marginBottom: 24 }}
      />
      
      <Tabs
        items={[
          {
            key: 'image',
            label: <span><PictureOutlined />图像搜索</span>,
            children: (
              <Card>
                <Upload.Dragger>
                  <p>上传图像进行相似度搜索</p>
                </Upload.Dragger>
              </Card>
            )
          },
          {
            key: 'audio', 
            label: <span><AudioOutlined />音频搜索</span>,
            children: (
              <Card>
                <p>音频向量搜索功能开发中...</p>
              </Card>
            )
          },
          {
            key: 'cross',
            label: <span><FileTextOutlined />跨模态</span>,
            children: (
              <Card>
                <Input.TextArea placeholder="输入文本描述，搜索相关图像" rows={4} />
                <Button type="primary" style={{ marginTop: 16 }}>
                  跨模态搜索
                </Button>
              </Card>
            )
          }
        ]}
      />
    </div>
  );
};

export default MultiModalSearchPanel;