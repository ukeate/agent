import React, { useEffect, useState } from 'react'
import { Card } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Alert } from '../components/ui/alert'
import { Input } from '../components/ui/input'
import { Textarea } from '../components/ui/textarea'
import { buildApiUrl, apiFetch } from '../utils/apiBase'

type SystemStatus = {
  status: string
  mode?: string
  active_sessions?: number
  privacy_level?: string
  updated_at?: string
}

type AnalysisResult = {
  success: boolean
  data?: any
  error?: any
}

const SocialIntelligenceDecisionPage: React.FC = () => {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [text, setText] = useState('')
  const [result, setResult] = useState<AnalysisResult | null>(null)

  const fetchJson = async <T,>(
    url: string,
    options?: RequestInit
  ): Promise<T> => {
    const res = await apiFetch(buildApiUrl(url), {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    })
    return res.json()
  }

  const loadStatus = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchJson<SystemStatus>(
        '/api/v1/social-emotion/status'
      )
      setStatus(data)
    } catch (e: any) {
      setError(e?.message || '加载失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadStatus()
  }, [])

  const analyze = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const payload = {
        user_id: 'demo-user',
        session_id: 'demo-session',
        emotion_data: { text },
        social_context: {},
        analysis_type: ['context_adaptation'],
        privacy_consent: true,
      }
      const data = await fetchJson<AnalysisResult>(
        '/api/v1/social-emotion/analyze',
        {
          method: 'POST',
          body: JSON.stringify(payload),
        }
      )
      setResult(data)
    } catch (e: any) {
      setError(e?.message || '分析失败')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">社交情感决策</h2>
        <Button onClick={loadStatus} disabled={loading}>
          刷新状态
        </Button>
      </div>

      {error && <Alert variant="destructive">{error}</Alert>}

      <Card className="p-4 space-y-2">
        <div className="flex items-center justify-between">
          <div className="font-medium">系统状态</div>
          <div className="text-sm text-gray-500">{status?.updated_at}</div>
        </div>
        <div className="text-sm text-gray-700">
          状态: {status?.status || '未知'}
          <br />
          模式: {status?.mode || '-'}
          <br />
          隐私级别: {status?.privacy_level || '-'}
          <br />
          活跃会话: {status?.active_sessions ?? '-'}
        </div>
      </Card>

      <Card className="p-4 space-y-3">
        <div className="font-medium">即时分析</div>
        <Textarea
          name="socialEmotionText"
          placeholder="输入待分析的文本或上下文"
          value={text}
          onChange={e => setText(e.target.value)}
          rows={4}
        />
        <div className="flex gap-2">
          <Button onClick={analyze} disabled={loading || !text.trim()}>
            提交分析
          </Button>
          <Button
            variant="outline"
            onClick={() => {
              setText('')
              setResult(null)
            }}
          >
            清空
          </Button>
        </div>
        {result && (
          <pre className="bg-gray-50 text-sm p-3 rounded border overflow-auto max-h-64">
            {JSON.stringify(result, null, 2)}
          </pre>
        )}
      </Card>
    </div>
  )
}

export default SocialIntelligenceDecisionPage
