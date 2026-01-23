import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState } from 'react'
import { Card, Input, Button, Alert, Space } from 'antd'

const SparqlQueryInterface: React.FC = () => {
  const [query, setQuery] = useState('SELECT * WHERE { ?s ?p ?o } LIMIT 10')
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const runQuery = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/kg/sparql/query'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      })
      const data = await res.json()
      setResult(data)
    } catch (e: any) {
      setError(e?.message || '查询失败')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Card title="SPARQL 查询">
          <Input.TextArea
            value={query}
            onChange={e => setQuery(e.target.value)}
            rows={6}
            placeholder="输入 SPARQL 查询语句"
          />
          <div style={{ marginTop: 12, textAlign: 'right' }}>
            <Button type="primary" onClick={runQuery} loading={loading}>
              执行
            </Button>
          </div>
        </Card>

        {error && <Alert type="error" message={error} />}
        {result && (
          <Card title="结果">
            <pre style={{ whiteSpace: 'pre-wrap' }}>
              {JSON.stringify(result, null, 2)}
            </pre>
          </Card>
        )}
      </Space>
    </div>
  )
}

export default SparqlQueryInterface
