"""pgvector 0.8 upgrade with quantization support

Revision ID: pgvector_0_8_upgrade
Revises: 
Create Date: 2025-08-15 10:00:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.dialects import postgresql
import uuid
from datetime import datetime
from src.core.utils.timezone_utils import utc_now

from src.core.logging import get_logger
logger = get_logger(__name__)

# revision identifiers
revision = 'pgvector_0_8_upgrade'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    """升级到pgvector 0.8"""
    
    # 获取数据库连接
    connection = op.get_bind()
    
    try:
        # 1. 确保pgvector扩展已安装
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        
        # 2. 升级pgvector扩展到0.8版本
        try:
            connection.execute(text("ALTER EXTENSION vector UPDATE TO '0.8'"))
        except Exception as e:
            logger.warning("升级vector扩展到0.8失败", error=str(e))
            # 继续执行，可能已经是0.8版本
        
        # 3. 创建量化参数表
        op.create_table(
            'vector_quantization_params',
            sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
            sa.Column('vector_id', postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column('table_name', sa.String(100), nullable=False),
            sa.Column('quantization_mode', sa.String(20), nullable=False),
            sa.Column('scale', sa.Float(), nullable=True),
            sa.Column('zero_point', sa.Integer(), nullable=True),
            sa.Column('centroids', sa.JSON(), nullable=True),
            sa.Column('compression_ratio', sa.Float(), nullable=False),
            sa.Column('precision_loss', sa.Float(), nullable=False),
            sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, default=utc_now),
            sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, default=utc_now)
        )
        
        # 4. 创建向量性能统计表
        op.create_table(
            'vector_performance_stats',
            sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
            sa.Column('table_name', sa.String(100), nullable=False),
            sa.Column('index_type', sa.String(20), nullable=False),
            sa.Column('search_latency_ms', sa.Float(), nullable=False),
            sa.Column('result_count', sa.Integer(), nullable=False),
            sa.Column('cache_hit', sa.Boolean(), nullable=False),
            sa.Column('timestamp', sa.TIMESTAMP(timezone=True), nullable=False, default=utc_now)
        )
        
        # 5. 创建知识库条目表
        op.create_table(
            'knowledge_items',
            sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
            sa.Column('content', sa.Text(), nullable=False),
            sa.Column('metadata', sa.JSON(), nullable=True, default={}),
            sa.Column('embedding', sa.Text(), nullable=True),  # Will be converted to VECTOR type
            sa.Column('embedding_quantized', sa.LargeBinary(), nullable=True),
            sa.Column('quantization_params_id', postgresql.UUID(as_uuid=True), nullable=True),
            sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, default=utc_now),
            sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, default=utc_now)
        )
        
        # 6. 修改embedding列为VECTOR类型
        try:
            connection.execute(text("ALTER TABLE knowledge_items ALTER COLUMN embedding TYPE vector(1536)"))
        except Exception as e:
            logger.warning("embedding列转换为vector类型失败", error=str(e))
        
        # 7. 创建索引
        op.create_index(
            'idx_vector_quantization_params_vector_id',
            'vector_quantization_params',
            ['vector_id']
        )
        op.create_index(
            'idx_vector_performance_stats_timestamp',
            'vector_performance_stats',
            ['timestamp']
        )
        op.create_index(
            'idx_knowledge_items_created_at',
            'knowledge_items',
            ['created_at']
        )
        
        # 8. 创建外键约束
        op.create_foreign_key(
            'fk_knowledge_items_quantization_params',
            'knowledge_items',
            'vector_quantization_params',
            ['quantization_params_id'],
            ['id']
        )
        
        # 9. 创建更新时间触发器
        connection.execute(text("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        """))
        
        # 为所有表创建触发器
        for table in ['vector_quantization_params', 'knowledge_items']:
            connection.execute(text(f"""
                DROP TRIGGER IF EXISTS update_{table}_updated_at ON {table};
                CREATE TRIGGER update_{table}_updated_at
                    BEFORE UPDATE ON {table}
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column();
            """))
        
        # 10. 尝试创建优化的HNSW索引（如果数据存在）
        try:
            connection.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_items_embedding_hnsw
                ON knowledge_items 
                USING hnsw (embedding vector_l2_ops)
                WITH (m = 16, ef_construction = 200)
            """))
        except Exception as e:
            logger.info("HNSW索引稍后创建", error=str(e))
        
        # 11. 优化PostgreSQL配置（会话级别）
        try:
            connection.execute(text("SET max_parallel_workers_per_gather = 4"))
            connection.execute(text("SET effective_cache_size = '2GB'"))
            connection.execute(text("SET random_page_cost = 1.1"))
        except Exception as e:
            logger.info("配置变更可能需要重启数据库", error=str(e))
        
        logger.info("pgvector 0.8 升级完成")
        
    except Exception as e:
        logger.exception("pgvector 升级失败")
        raise

def downgrade():
    """降级pgvector版本"""
    
    connection = op.get_bind()
    
    try:
        # 删除外键约束
        op.drop_constraint('fk_knowledge_items_quantization_params', 'knowledge_items')
        
        # 删除索引
        op.drop_index('idx_knowledge_items_created_at', 'knowledge_items')
        op.drop_index('idx_vector_performance_stats_timestamp', 'vector_performance_stats')
        op.drop_index('idx_vector_quantization_params_vector_id', 'vector_quantization_params')
        
        # 删除HNSW索引
        try:
            connection.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_knowledge_items_embedding_hnsw"))
        except Exception as e:
            logger.info("删除HNSW索引失败", error=str(e))
        
        # 删除触发器
        for table in ['vector_quantization_params', 'knowledge_items']:
            try:
                connection.execute(text(f"DROP TRIGGER IF EXISTS update_{table}_updated_at ON {table}"))
            except Exception as e:
                logger.info("删除触发器失败", table=table, error=str(e))
        
        # 删除触发器函数
        try:
            connection.execute(text("DROP FUNCTION IF EXISTS update_updated_at_column()"))
        except Exception as e:
            logger.info("删除触发器函数失败", error=str(e))
        
        # 删除表
        op.drop_table('knowledge_items')
        op.drop_table('vector_performance_stats')
        op.drop_table('vector_quantization_params')
        
        # 降级扩展版本（谨慎操作）
        try:
            connection.execute(text("ALTER EXTENSION vector UPDATE TO '0.5'"))
        except Exception as e:
            logger.warning("降级vector扩展失败", error=str(e))
        
        logger.info("pgvector 降级完成")
        
    except Exception as e:
        logger.exception("pgvector 降级失败")
        raise
