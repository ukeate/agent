-- pgvector 0.8.0 初始化和优化配置
-- 启用pgvector扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 检查pgvector版本
SELECT extversion FROM pg_extension WHERE extname = 'vector';

-- 创建pgvector优化配置函数
CREATE OR REPLACE FUNCTION configure_pgvector_performance() RETURNS void AS $$
BEGIN
    -- HNSW索引优化配置
    PERFORM set_config('hnsw.ef_search', '100', false);
    PERFORM set_config('hnsw.iterative_scan', 'strict_order', false);
    PERFORM set_config('hnsw.max_scan_tuples', '20000', false);
    PERFORM set_config('hnsw.scan_mem_multiplier', '2', false);
    
    -- IVFFlat索引优化配置
    PERFORM set_config('ivfflat.probes', '10', false);
    PERFORM set_config('ivfflat.iterative_scan', 'strict_order', false);
    PERFORM set_config('ivfflat.max_probes', '100', false);
    
    -- 并行处理优化
    PERFORM set_config('max_parallel_workers_per_gather', '4', false);
    PERFORM set_config('min_parallel_table_scan_size', '8MB', false);
    PERFORM set_config('parallel_setup_cost', '1000', false);
    PERFORM set_config('parallel_tuple_cost', '0.1', false);
    
    -- 查询规划器优化
    PERFORM set_config('random_page_cost', '1.1', false);
    PERFORM set_config('seq_page_cost', '1.0', false);
    PERFORM set_config('cpu_tuple_cost', '0.01', false);
    PERFORM set_config('cpu_index_tuple_cost', '0.005', false);
    PERFORM set_config('cpu_operator_cost', '0.0025', false);
    
    RAISE NOTICE 'pgvector性能配置已应用';
END;
$$ LANGUAGE plpgsql;

-- 应用优化配置
SELECT configure_pgvector_performance();

-- 创建向量索引监控视图
CREATE OR REPLACE VIEW vector_index_stats AS
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
    pg_relation_size(indexrelid) as index_size_bytes
FROM pg_stat_user_indexes 
WHERE indexname LIKE '%_embedding_%' 
   OR indexname LIKE '%vector%'
ORDER BY pg_relation_size(indexrelid) DESC;

-- 创建向量查询性能监控函数
CREATE OR REPLACE FUNCTION get_vector_query_stats()
RETURNS TABLE (
    query_text text,
    calls bigint,
    total_time_ms numeric,
    mean_time_ms numeric,
    index_usage text
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pss.query,
        pss.calls,
        round((pss.total_plan_time + pss.total_exec_time)::numeric, 2) as total_time_ms,
        round(((pss.total_plan_time + pss.total_exec_time) / pss.calls)::numeric, 2) as mean_time_ms,
        CASE 
            WHEN pss.query LIKE '%<->%' OR pss.query LIKE '%<#>%' OR pss.query LIKE '%<=>%' OR pss.query LIKE '%<+>%' 
            THEN 'vector_similarity'
            ELSE 'other'
        END as index_usage
    FROM pg_stat_statements pss
    WHERE pss.query LIKE '%embedding%' 
       OR pss.query LIKE '%vector%'
       OR pss.query LIKE '%<->%'
       OR pss.query LIKE '%<#>%'  
       OR pss.query LIKE '%<=>%'
       OR pss.query LIKE '%<+>%'
    ORDER BY (pss.total_plan_time + pss.total_exec_time) DESC
    LIMIT 20;
END;
$$ LANGUAGE plpgsql;

-- 启用pg_stat_statements用于查询性能监控
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- 创建向量量化支持函数
CREATE OR REPLACE FUNCTION create_optimized_vector_index(
    table_name text,
    column_name text,
    index_type text DEFAULT 'hnsw',
    distance_metric text DEFAULT 'l2',
    dimensions integer DEFAULT NULL,
    index_options text DEFAULT ''
) RETURNS text AS $$
DECLARE
    sql_statement text;
    ops_class text;
    full_table_name text;
BEGIN
    -- 确定操作符类
    CASE distance_metric
        WHEN 'l2' THEN ops_class := 'vector_l2_ops';
        WHEN 'cosine' THEN ops_class := 'vector_cosine_ops';
        WHEN 'ip' THEN ops_class := 'vector_ip_ops';
        WHEN 'l1' THEN ops_class := 'vector_l1_ops';
        ELSE 
            RAISE EXCEPTION '不支持的距离度量: %', distance_metric;
    END CASE;
    
    -- 构建索引创建语句
    full_table_name := quote_ident(table_name);
    
    IF index_type = 'hnsw' THEN
        sql_statement := format(
            'CREATE INDEX CONCURRENTLY IF NOT EXISTS %I ON %s USING hnsw (%I %s) %s',
            table_name || '_' || column_name || '_hnsw_idx',
            full_table_name,
            column_name,
            ops_class,
            COALESCE('WITH (' || index_options || ')', '')
        );
    ELSIF index_type = 'ivfflat' THEN
        sql_statement := format(
            'CREATE INDEX CONCURRENTLY IF NOT EXISTS %I ON %s USING ivfflat (%I %s) %s',
            table_name || '_' || column_name || '_ivfflat_idx', 
            full_table_name,
            column_name,
            ops_class,
            COALESCE('WITH (' || index_options || ')', '')
        );
    ELSE
        RAISE EXCEPTION '不支持的索引类型: %', index_type;
    END IF;
    
    EXECUTE sql_statement;
    
    RETURN format('索引创建成功: %s', sql_statement);
EXCEPTION
    WHEN others THEN
        RETURN format('索引创建失败: %s, 错误: %s', sql_statement, SQLERRM);
END;
$$ LANGUAGE plpgsql;

-- 记录初始化完成
INSERT INTO public.system_logs (created_at, level, message, details) 
VALUES (
    NOW(), 
    'INFO', 
    'pgvector 0.8.0初始化完成',
    jsonb_build_object(
        'version', (SELECT extversion FROM pg_extension WHERE extname = 'vector'),
        'optimization_applied', true,
        'monitoring_enabled', true
    )
) ON CONFLICT DO NOTHING;

-- 创建system_logs表如果不存在
CREATE TABLE IF NOT EXISTS public.system_logs (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    level VARCHAR(10) NOT NULL,
    message TEXT NOT NULL,
    details JSONB
);