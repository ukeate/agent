import React, { useEffect, useState } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Alert } from '../components/ui/alert';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/ui/tabs';
import { behaviorAnalyticsService } from '../services/behaviorAnalyticsService';
import { BehaviorOverview } from '../components/analytics/BehaviorOverview';
import { PatternAnalysis } from '../components/analytics/PatternAnalysis';
import { AnomalyDetection } from '../components/analytics/AnomalyDetection';
import { TrendAnalysis } from '../components/analytics/TrendAnalysis';
import { RealTimeMonitor } from '../components/analytics/RealTimeMonitor';
import { PerformanceMetrics } from '../components/analytics/PerformanceMetrics';

type AnalyticsData = {
  overview: any;
  patterns: any[];
  anomalies: any[];
  trends: any;
  realtime: any;
  performance: any;
};

const BehaviorAnalyticsPage: React.FC = () => {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [realtimeEvents, setRealtimeEvents] = useState<any[]>([]);
  const [realtimeConnected, setRealtimeConnected] = useState(false);

  const computeOverview = (events: any[]) => {
    const total = events.length;
    const uniqueUsers = new Set(events.map(e => e.user_id)).size;
    const uniqueSessions = new Set(events.map(e => e.session_id)).size;

    const eventTypeDist: Record<string, number> = {};
    const hourly: Record<string, number> = {};
    let minTs: number | null = null;
    let maxTs: number | null = null;
    events.forEach(e => {
      const t = e.timestamp ? new Date(e.timestamp) : null;
      const type = e.event_type || 'unknown';
      eventTypeDist[type] = (eventTypeDist[type] || 0) + 1;
      if (t) {
        const ms = t.getTime();
        if (!Number.isNaN(ms)) {
          minTs = minTs === null ? ms : Math.min(minTs, ms);
          maxTs = maxTs === null ? ms : Math.max(maxTs, ms);
        }
        const h = String(t.getHours());
        hourly[h] = (hourly[h] || 0) + 1;
      }
    });

    const eventsPerMinute =
      minTs !== null && maxTs !== null && maxTs > minTs
        ? total / ((maxTs - minTs) / 60000)
        : 0;

    return {
      total_events: total,
      unique_users: uniqueUsers,
      unique_sessions: uniqueSessions,
      events_per_minute: eventsPerMinute,
      event_type_distribution: eventTypeDist,
      hourly_distribution: hourly,
      most_active_hour: Number(
        Object.entries(hourly).sort(([, a], [, b]) => (b as number) - (a as number))[0]?.[0]
      )
    };
  };

  const computeRealtimeStats = (events: any[]) => {
    const windowDurationSeconds = 300;
    const now = Date.now();
    const windowMs = windowDurationSeconds * 1000;
    const windowEvents = events.filter(e => {
      if (!e.timestamp) return false;
      const ms = new Date(e.timestamp).getTime();
      return !Number.isNaN(ms) && now - ms <= windowMs;
    });

    const eventTypeDist: Record<string, number> = {};
    const hourly: Record<string, number> = {};
    windowEvents.forEach(e => {
      const type = e.event_type || 'unknown';
      eventTypeDist[type] = (eventTypeDist[type] || 0) + 1;
      const t = e.timestamp ? new Date(e.timestamp) : null;
      if (t) {
        const h = String(t.getHours());
        hourly[h] = (hourly[h] || 0) + 1;
      }
    });

    const eventCount = windowEvents.length;
    const activeUsers = new Set(windowEvents.map(e => e.user_id).filter(Boolean)).size;
    const uniqueSessions = new Set(windowEvents.map(e => e.session_id).filter(Boolean)).size;

    return {
      event_count: eventCount,
      active_users: activeUsers,
      events_per_minute: eventCount / (windowDurationSeconds / 60),
      window_duration_seconds: windowDurationSeconds,
      unique_sessions: uniqueSessions,
      event_type_distribution: eventTypeDist,
      hourly_distribution: hourly,
      most_active_hour: Number(
        Object.entries(hourly).sort(([, a], [, b]) => (b as number) - (a as number))[0]?.[0]
      )
    };
  };

  const realtimeStats = computeRealtimeStats(realtimeEvents);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [eventsResp, anomaliesResp, patternsResp, wsStatsResp] = await Promise.all([
        behaviorAnalyticsService.getEvents({ limit: 500 }),
        behaviorAnalyticsService.getAnomalies({ limit: 100 }),
        behaviorAnalyticsService.getPatterns({ limit: 100 }),
        behaviorAnalyticsService.getWebSocketStats()
      ]);

      const events = eventsResp?.events || [];
      const overview = computeOverview(events);

      setData({
        overview,
        patterns: patternsResp?.patterns || [],
        anomalies: anomaliesResp?.anomalies || [],
        trends: {},
        realtime: {},
        performance: wsStatsResp?.stats || {}
      });
    } catch (e: any) {
      setError(e?.message || '加载失败');
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    const unsub = behaviorAnalyticsService.subscribeToRealtimeEvents(
      (event) => {
        setRealtimeEvents(prev => [event, ...prev.slice(0, 199)]);
      },
      (connected) => setRealtimeConnected(connected)
    );
    return () => {
      if (unsub) unsub();
    };
  }, []);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">行为分析</h2>
        <Button onClick={loadData} disabled={loading}>刷新</Button>
      </div>

      {error && <Alert variant="destructive">{error}</Alert>}

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="overview">概览</TabsTrigger>
          <TabsTrigger value="patterns">模式</TabsTrigger>
          <TabsTrigger value="anomalies">异常</TabsTrigger>
          <TabsTrigger value="trends">趋势</TabsTrigger>
          <TabsTrigger value="realtime">实时</TabsTrigger>
          <TabsTrigger value="performance">性能</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <Card className="p-4">
            {data ? (
              <BehaviorOverview data={data.overview} realtime={realtimeStats} />
            ) : (
              <div className="text-sm text-gray-500">暂无数据</div>
            )}
          </Card>
        </TabsContent>

        <TabsContent value="patterns">
          <PatternAnalysis patterns={data?.patterns || []} loading={loading} onRefresh={loadData} />
        </TabsContent>

        <TabsContent value="anomalies">
          <AnomalyDetection anomalies={data?.anomalies || []} loading={loading} onRefresh={loadData} />
        </TabsContent>

        <TabsContent value="trends">
          <TrendAnalysis trendData={data?.trends || {}} loading={loading} />
        </TabsContent>

        <TabsContent value="realtime">
          <RealTimeMonitor events={realtimeEvents} stats={realtimeStats} connected={realtimeConnected} />
        </TabsContent>

        <TabsContent value="performance">
          <PerformanceMetrics metrics={data?.performance || {}} />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default BehaviorAnalyticsPage;
