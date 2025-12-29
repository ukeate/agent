import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState } from 'react';
import { Card, Input, Button, Alert, Select, Space } from 'antd';

const ModelInferencePage: React.FC = () => {
  const [model, setModel] = useState('');
  const [prompt, setPrompt] = useState('');
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/model-service/infer'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: model), input: prompt })
      });
      const data = await res.json();
      setResult(data);
    } catch (e: any) {
      setError(e?.message || '推理失败');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Card title="模型推理">
          <Space style={{ width: '100%' }} direction="vertical">
            <Select
              placeholder="选择模型ID"
              value={model || undefined}
              onChange={setModel}
              options={[]}
              style={{ width: 320 }}
            />
            <Input.TextArea
              rows={4}
              placeholder="输入推理提示"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
            <Button type="primary" onClick={run} disabled={!model || !prompt.trim()} loading={loading}>
              执行
            </Button>
          </Space>
        </Card>

        {error && <Alert type="error" message={error} />}
        {result && (
          <Card title="结果">
            <pre style={{ whiteSpace: 'pre-wrap' }}>{JSON.stringify(result, null, 2)}</pre>
          </Card>
        )}
      </Space>
    </div>
  );
};

export default ModelInferencePage;
