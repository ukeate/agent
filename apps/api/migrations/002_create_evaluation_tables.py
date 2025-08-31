"""
模型评估系统数据库迁移脚本

创建所有与模型评估相关的表结构
"""

import asyncio
import asyncpg
from datetime import datetime
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# 表结构定义
TABLES = {
    "model_info": """
        CREATE TABLE IF NOT EXISTS model_info (
            id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            version VARCHAR(100),
            description TEXT,
            model_path TEXT NOT NULL,
            model_type VARCHAR(100) DEFAULT 'text_generation',
            architecture VARCHAR(100),
            parameters_count BIGINT,
            training_data TEXT,
            license_info TEXT,
            created_by VARCHAR(255) DEFAULT 'system',
            tags JSONB DEFAULT '[]',
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_model_info_name ON model_info(name);
        CREATE INDEX IF NOT EXISTS idx_model_info_type ON model_info(model_type);
        CREATE INDEX IF NOT EXISTS idx_model_info_created_by ON model_info(created_by);
        CREATE INDEX IF NOT EXISTS idx_model_info_tags ON model_info USING GIN(tags);
    """,
    
    "benchmark_definition": """
        CREATE TABLE IF NOT EXISTS benchmark_definition (
            id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            display_name VARCHAR(255) NOT NULL,
            description TEXT NOT NULL,
            category VARCHAR(100) DEFAULT 'general',
            difficulty VARCHAR(50) DEFAULT 'medium',
            tasks JSONB DEFAULT '[]',
            languages JSONB DEFAULT '["en"]',
            metrics JSONB DEFAULT '[]',
            num_samples INTEGER,
            estimated_runtime_minutes INTEGER,
            memory_requirements_gb REAL,
            requires_gpu BOOLEAN DEFAULT FALSE,
            dataset_name VARCHAR(255),
            dataset_version VARCHAR(100),
            dataset_split VARCHAR(50) DEFAULT 'test',
            paper_url TEXT,
            homepage_url TEXT,
            citation TEXT,
            license_info TEXT,
            few_shot_examples INTEGER DEFAULT 0,
            prompt_template TEXT,
            evaluation_script TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            created_by VARCHAR(255) DEFAULT 'system',
            tags JSONB DEFAULT '[]',
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_benchmark_name ON benchmark_definition(name);
        CREATE INDEX IF NOT EXISTS idx_benchmark_category ON benchmark_definition(category);
        CREATE INDEX IF NOT EXISTS idx_benchmark_difficulty ON benchmark_definition(difficulty);
        CREATE INDEX IF NOT EXISTS idx_benchmark_active ON benchmark_definition(is_active);
        CREATE INDEX IF NOT EXISTS idx_benchmark_tags ON benchmark_definition USING GIN(tags);
    """,
    
    "evaluation_job": """
        CREATE TABLE IF NOT EXISTS evaluation_job (
            id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            status VARCHAR(50) DEFAULT 'pending',
            created_by VARCHAR(255) DEFAULT 'system',
            models JSONB DEFAULT '[]',
            benchmarks JSONB DEFAULT '[]',
            progress REAL DEFAULT 0.0,
            current_task TEXT,
            started_at TIMESTAMP WITH TIME ZONE,
            completed_at TIMESTAMP WITH TIME ZONE,
            results JSONB DEFAULT '[]',
            summary_metrics JSONB DEFAULT '{}',
            error_message TEXT,
            estimated_duration_minutes INTEGER,
            actual_duration_seconds REAL,
            peak_memory_usage_gb REAL,
            total_samples_processed INTEGER DEFAULT 0,
            evaluation_config JSONB DEFAULT '{}',
            hardware_info JSONB DEFAULT '{}',
            tags JSONB DEFAULT '[]',
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_evaluation_job_status ON evaluation_job(status);
        CREATE INDEX IF NOT EXISTS idx_evaluation_job_created_by ON evaluation_job(created_by);
        CREATE INDEX IF NOT EXISTS idx_evaluation_job_created_at ON evaluation_job(created_at);
        CREATE INDEX IF NOT EXISTS idx_evaluation_job_tags ON evaluation_job USING GIN(tags);
    """,
    
    "evaluation_result": """
        CREATE TABLE IF NOT EXISTS evaluation_result (
            id VARCHAR(255) PRIMARY KEY,
            job_id VARCHAR(255) NOT NULL,
            model_name VARCHAR(255) NOT NULL,
            benchmark_name VARCHAR(255) NOT NULL,
            task_name VARCHAR(255) NOT NULL,
            accuracy REAL DEFAULT 0.0,
            f1_score REAL,
            precision REAL,
            recall REAL,
            bleu_score REAL,
            rouge_scores JSONB DEFAULT '{}',
            perplexity REAL,
            inference_time_ms REAL DEFAULT 0.0,
            memory_usage_mb REAL DEFAULT 0.0,
            throughput_samples_per_sec REAL DEFAULT 0.0,
            gpu_utilization_percent REAL,
            samples_evaluated INTEGER DEFAULT 0,
            execution_time_seconds REAL DEFAULT 0.0,
            success BOOLEAN DEFAULT TRUE,
            error_message TEXT,
            detailed_metrics JSONB DEFAULT '{}',
            sample_outputs JSONB DEFAULT '[]',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES evaluation_job(id) ON DELETE CASCADE
        );
        
        CREATE INDEX IF NOT EXISTS idx_evaluation_result_job_id ON evaluation_result(job_id);
        CREATE INDEX IF NOT EXISTS idx_evaluation_result_model ON evaluation_result(model_name);
        CREATE INDEX IF NOT EXISTS idx_evaluation_result_benchmark ON evaluation_result(benchmark_name);
        CREATE INDEX IF NOT EXISTS idx_evaluation_result_accuracy ON evaluation_result(accuracy);
        CREATE INDEX IF NOT EXISTS idx_evaluation_result_success ON evaluation_result(success);
        CREATE INDEX IF NOT EXISTS idx_evaluation_result_created_at ON evaluation_result(created_at);
    """,
    
    "performance_metric": """
        CREATE TABLE IF NOT EXISTS performance_metric (
            id VARCHAR(255) PRIMARY KEY,
            job_id VARCHAR(255),
            model_name VARCHAR(255),
            benchmark_name VARCHAR(255),
            metric_type VARCHAR(50) DEFAULT 'system',
            metric_name VARCHAR(255) NOT NULL,
            metric_value REAL NOT NULL,
            metric_unit VARCHAR(50) DEFAULT '',
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            duration_seconds REAL,
            context JSONB DEFAULT '{}',
            tags JSONB DEFAULT '[]'
        );
        
        CREATE INDEX IF NOT EXISTS idx_performance_metric_job_id ON performance_metric(job_id);
        CREATE INDEX IF NOT EXISTS idx_performance_metric_type ON performance_metric(metric_type);
        CREATE INDEX IF NOT EXISTS idx_performance_metric_name ON performance_metric(metric_name);
        CREATE INDEX IF NOT EXISTS idx_performance_metric_timestamp ON performance_metric(timestamp);
        CREATE INDEX IF NOT EXISTS idx_performance_metric_model ON performance_metric(model_name);
        CREATE INDEX IF NOT EXISTS idx_performance_metric_tags ON performance_metric USING GIN(tags);
    """,
    
    "performance_alert": """
        CREATE TABLE IF NOT EXISTS performance_alert (
            id VARCHAR(255) PRIMARY KEY,
            alert_type VARCHAR(50) DEFAULT 'performance',
            severity VARCHAR(50) DEFAULT 'medium',
            title VARCHAR(255) NOT NULL,
            description TEXT NOT NULL,
            metric_name VARCHAR(255) NOT NULL,
            current_value REAL NOT NULL,
            threshold_value REAL NOT NULL,
            job_id VARCHAR(255),
            model_name VARCHAR(255),
            benchmark_name VARCHAR(255),
            is_resolved BOOLEAN DEFAULT FALSE,
            resolved_at TIMESTAMP WITH TIME ZONE,
            resolved_by VARCHAR(255),
            resolution_note TEXT,
            is_auto_created BOOLEAN DEFAULT TRUE,
            auto_resolve_after_minutes INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_performance_alert_severity ON performance_alert(severity);
        CREATE INDEX IF NOT EXISTS idx_performance_alert_resolved ON performance_alert(is_resolved);
        CREATE INDEX IF NOT EXISTS idx_performance_alert_job_id ON performance_alert(job_id);
        CREATE INDEX IF NOT EXISTS idx_performance_alert_model ON performance_alert(model_name);
        CREATE INDEX IF NOT EXISTS idx_performance_alert_created_at ON performance_alert(created_at);
    """,
    
    "evaluation_report": """
        CREATE TABLE IF NOT EXISTS evaluation_report (
            id VARCHAR(255) PRIMARY KEY,
            title VARCHAR(255) NOT NULL,
            subtitle TEXT,
            status VARCHAR(50) DEFAULT 'generating',
            include_charts BOOLEAN DEFAULT TRUE,
            include_detailed_metrics BOOLEAN DEFAULT TRUE,
            include_recommendations BOOLEAN DEFAULT TRUE,
            output_format VARCHAR(50) DEFAULT 'html',
            template_name VARCHAR(100) DEFAULT 'default',
            job_ids JSONB DEFAULT '[]',
            model_names JSONB DEFAULT '[]',
            benchmark_names JSONB DEFAULT '[]',
            file_path TEXT,
            file_size_bytes BIGINT,
            download_count INTEGER DEFAULT 0,
            generated_by VARCHAR(255) DEFAULT 'system',
            generation_time_seconds REAL,
            error_message TEXT,
            expires_at TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_evaluation_report_status ON evaluation_report(status);
        CREATE INDEX IF NOT EXISTS idx_evaluation_report_generated_by ON evaluation_report(generated_by);
        CREATE INDEX IF NOT EXISTS idx_evaluation_report_created_at ON evaluation_report(created_at);
        CREATE INDEX IF NOT EXISTS idx_evaluation_report_expires_at ON evaluation_report(expires_at);
    """,
    
    "comparison_study": """
        CREATE TABLE IF NOT EXISTS comparison_study (
            id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            model_names JSONB DEFAULT '[]',
            benchmark_names JSONB DEFAULT '[]',
            comparison_metrics JSONB DEFAULT '[]',
            winner_model VARCHAR(255),
            key_findings JSONB DEFAULT '[]',
            recommendations JSONB DEFAULT '[]',
            statistical_significance JSONB DEFAULT '{}',
            confidence_intervals JSONB DEFAULT '{}',
            job_ids JSONB DEFAULT '[]',
            report_id VARCHAR(255),
            created_by VARCHAR(255) DEFAULT 'system',
            is_published BOOLEAN DEFAULT FALSE,
            tags JSONB DEFAULT '[]',
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_comparison_study_published ON comparison_study(is_published);
        CREATE INDEX IF NOT EXISTS idx_comparison_study_created_by ON comparison_study(created_by);
        CREATE INDEX IF NOT EXISTS idx_comparison_study_tags ON comparison_study USING GIN(tags);
    """,
    
    "baseline_model": """
        CREATE TABLE IF NOT EXISTS baseline_model (
            id VARCHAR(255) PRIMARY KEY,
            model_name VARCHAR(255) NOT NULL,
            benchmark_name VARCHAR(255) NOT NULL,
            baseline_accuracy REAL DEFAULT 0.0,
            baseline_f1_score REAL,
            baseline_inference_time_ms REAL DEFAULT 0.0,
            baseline_memory_usage_mb REAL DEFAULT 0.0,
            established_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            established_by VARCHAR(255) DEFAULT 'system',
            source_job_id VARCHAR(255) NOT NULL,
            performance_degradation_threshold REAL DEFAULT 0.05,
            speed_degradation_threshold REAL DEFAULT 0.20,
            memory_increase_threshold REAL DEFAULT 0.30,
            notes TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            tags JSONB DEFAULT '[]',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(model_name, benchmark_name)
        );
        
        CREATE INDEX IF NOT EXISTS idx_baseline_model_name ON baseline_model(model_name);
        CREATE INDEX IF NOT EXISTS idx_baseline_model_benchmark ON baseline_model(benchmark_name);
        CREATE INDEX IF NOT EXISTS idx_baseline_model_active ON baseline_model(is_active);
        CREATE INDEX IF NOT EXISTS idx_baseline_model_established ON baseline_model(established_date);
    """
}

# 触发器定义（用于自动更新updated_at字段）
TRIGGERS = {
    "model_info": """
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        
        DROP TRIGGER IF EXISTS update_model_info_updated_at ON model_info;
        CREATE TRIGGER update_model_info_updated_at
            BEFORE UPDATE ON model_info
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """,
    
    "benchmark_definition": """
        DROP TRIGGER IF EXISTS update_benchmark_definition_updated_at ON benchmark_definition;
        CREATE TRIGGER update_benchmark_definition_updated_at
            BEFORE UPDATE ON benchmark_definition
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """,
    
    "evaluation_job": """
        DROP TRIGGER IF EXISTS update_evaluation_job_updated_at ON evaluation_job;
        CREATE TRIGGER update_evaluation_job_updated_at
            BEFORE UPDATE ON evaluation_job
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """,
    
    "performance_alert": """
        DROP TRIGGER IF EXISTS update_performance_alert_updated_at ON performance_alert;
        CREATE TRIGGER update_performance_alert_updated_at
            BEFORE UPDATE ON performance_alert
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """,
    
    "evaluation_report": """
        DROP TRIGGER IF EXISTS update_evaluation_report_updated_at ON evaluation_report;
        CREATE TRIGGER update_evaluation_report_updated_at
            BEFORE UPDATE ON evaluation_report
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """,
    
    "comparison_study": """
        DROP TRIGGER IF EXISTS update_comparison_study_updated_at ON comparison_study;
        CREATE TRIGGER update_comparison_study_updated_at
            BEFORE UPDATE ON comparison_study
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """,
    
    "baseline_model": """
        DROP TRIGGER IF EXISTS update_baseline_model_updated_at ON baseline_model;
        CREATE TRIGGER update_baseline_model_updated_at
            BEFORE UPDATE ON baseline_model
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """
}

# 初始数据
INITIAL_DATA = {
    "benchmark_definition": [
        {
            "id": "glue-cola",
            "name": "cola",
            "display_name": "CoLA",
            "description": "Corpus of Linguistic Acceptability - 语言可接受性语料库",
            "category": "nlp",
            "difficulty": "medium",
            "tasks": ["cola"],
            "languages": ["en"],
            "metrics": ["accuracy", "f1"],
            "num_samples": 1043,
            "estimated_runtime_minutes": 5,
            "memory_requirements_gb": 2.0,
            "requires_gpu": False,
            "dataset_name": "glue",
            "dataset_version": "1.0.0",
            "few_shot_examples": 0,
            "paper_url": "https://nyu-mll.github.io/CoLA/",
            "tags": ["glue", "grammar", "acceptability"]
        },
        {
            "id": "glue-sst2", 
            "name": "sst2",
            "display_name": "SST-2",
            "description": "Stanford Sentiment Treebank - 斯坦福情感树库",
            "category": "nlp",
            "difficulty": "easy",
            "tasks": ["sst2"],
            "languages": ["en"],
            "metrics": ["accuracy"],
            "num_samples": 1821,
            "estimated_runtime_minutes": 8,
            "memory_requirements_gb": 2.0,
            "requires_gpu": False,
            "dataset_name": "glue",
            "dataset_version": "1.0.0",
            "tags": ["glue", "sentiment", "classification"]
        },
        {
            "id": "mmlu",
            "name": "mmlu", 
            "display_name": "MMLU",
            "description": "Massive Multitask Language Understanding - 大规模多任务语言理解",
            "category": "knowledge",
            "difficulty": "hard",
            "tasks": ["mmlu"],
            "languages": ["en"],
            "metrics": ["accuracy"],
            "num_samples": 14042,
            "estimated_runtime_minutes": 60,
            "memory_requirements_gb": 8.0,
            "requires_gpu": True,
            "dataset_name": "cais/mmlu",
            "dataset_version": "all",
            "paper_url": "https://arxiv.org/abs/2009.03300",
            "tags": ["knowledge", "multitask", "reasoning"]
        },
        {
            "id": "hellaswag",
            "name": "hellaswag",
            "display_name": "HellaSwag",
            "description": "Can a Machine Really Finish Your Sentence? - 机器能否真正完成你的句子？",
            "category": "reasoning",
            "difficulty": "hard", 
            "tasks": ["hellaswag"],
            "languages": ["en"],
            "metrics": ["accuracy"],
            "num_samples": 10042,
            "estimated_runtime_minutes": 25,
            "memory_requirements_gb": 4.0,
            "requires_gpu": True,
            "dataset_name": "hellaswag",
            "paper_url": "https://arxiv.org/abs/1905.07830",
            "tags": ["reasoning", "commonsense", "completion"]
        },
        {
            "id": "humaneval",
            "name": "humaneval",
            "display_name": "HumanEval", 
            "description": "Evaluating Large Language Models Trained on Code - 评估在代码上训练的大型语言模型",
            "category": "code",
            "difficulty": "hard",
            "tasks": ["humaneval"],
            "languages": ["python"],
            "metrics": ["pass@1", "pass@10", "pass@100"],
            "num_samples": 164,
            "estimated_runtime_minutes": 30,
            "memory_requirements_gb": 6.0,
            "requires_gpu": True,
            "dataset_name": "openai_humaneval",
            "paper_url": "https://arxiv.org/abs/2107.03374",
            "tags": ["code", "programming", "python"]
        }
    ]
}

async def run_migration(database_url: str):
    """执行数据库迁移"""
    try:
        logger.info("开始执行模型评估系统数据库迁移...")
        
        # 连接数据库
        conn = await asyncpg.connect(database_url)
        
        try:
            # 创建表
            logger.info("创建数据表...")
            for table_name, table_sql in TABLES.items():
                logger.info(f"创建表: {table_name}")
                await conn.execute(table_sql)
            
            # 创建触发器
            logger.info("创建触发器...")
            for trigger_name, trigger_sql in TRIGGERS.items():
                logger.info(f"创建触发器: {trigger_name}")
                await conn.execute(trigger_sql)
            
            # 插入初始数据
            logger.info("插入初始数据...")
            for table_name, records in INITIAL_DATA.items():
                if records:
                    logger.info(f"插入数据到表: {table_name} ({len(records)} 条记录)")
                    
                    for record in records:
                        # 构建插入语句
                        columns = list(record.keys())
                        placeholders = [f"${i+1}" for i in range(len(columns))]
                        values = [record[col] for col in columns]
                        
                        # 处理JSON字段
                        for i, value in enumerate(values):
                            if isinstance(value, (list, dict)):
                                values[i] = str(value).replace("'", '"')  # 转为JSON格式字符串
                        
                        insert_sql = f"""
                            INSERT INTO {table_name} ({', '.join(columns)})
                            VALUES ({', '.join(placeholders)})
                            ON CONFLICT DO NOTHING
                        """
                        
                        await conn.execute(insert_sql, *values)
            
            logger.info("✅ 模型评估系统数据库迁移完成!")
            
        finally:
            await conn.close()
            
    except Exception as e:
        logger.error(f"❌ 数据库迁移失败: {e}")
        raise

async def rollback_migration(database_url: str):
    """回滚数据库迁移"""
    try:
        logger.info("开始回滚模型评估系统数据库迁移...")
        
        conn = await asyncpg.connect(database_url)
        
        try:
            # 删除表（按依赖顺序）
            tables_to_drop = [
                "evaluation_result",
                "performance_metric", 
                "performance_alert",
                "evaluation_report",
                "comparison_study",
                "baseline_model",
                "evaluation_job",
                "benchmark_definition",
                "model_info"
            ]
            
            for table_name in tables_to_drop:
                logger.info(f"删除表: {table_name}")
                await conn.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
            
            # 删除函数
            await conn.execute("DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE")
            
            logger.info("✅ 模型评估系统数据库迁移回滚完成!")
            
        finally:
            await conn.close()
            
    except Exception as e:
        logger.error(f"❌ 数据库迁移回滚失败: {e}")
        raise

if __name__ == "__main__":
    import os
    import sys
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 从环境变量获取数据库连接
    database_url = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/agent_db")
    
    # 根据参数决定执行还是回滚
    if len(sys.argv) > 1 and sys.argv[1] == "rollback":
        asyncio.run(rollback_migration(database_url))
    else:
        asyncio.run(run_migration(database_url))