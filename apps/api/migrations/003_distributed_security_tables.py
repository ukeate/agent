"""
分布式安全框架数据库表
创建支持身份认证、访问控制、安全审计的数据库表结构

Revision ID: 003_distributed_security
Revises: 002_create_evaluation_tables
Create Date: 2025-01-15 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '003_distributed_security'
down_revision = '002_create_evaluation_tables'
branch_labels = None
depends_on = None


def upgrade():
    """创建分布式安全框架数据库表"""
    
    # 1. 智能体身份表
    op.create_table(
        'agent_identities',
        sa.Column('agent_id', sa.String(255), primary_key=True),
        sa.Column('public_key', sa.Text, nullable=False),
        sa.Column('certificate', sa.Text, nullable=True),
        sa.Column('trust_level', sa.Integer, nullable=False, default=1),
        sa.Column('roles', postgresql.ARRAY(sa.String(100)), nullable=False, default=[]),
        sa.Column('attributes', postgresql.JSONB, nullable=True),
        sa.Column('issued_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('revoked', sa.Boolean, nullable=False, default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )
    
    # 创建索引
    op.create_index('idx_agent_identities_expires', 'agent_identities', ['expires_at'])
    op.create_index('idx_agent_identities_trust_level', 'agent_identities', ['trust_level'])
    op.create_index('idx_agent_identities_revoked', 'agent_identities', ['revoked'])
    
    # 2. 访问策略表
    op.create_table(
        'access_policies',
        sa.Column('policy_id', sa.String(255), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('target', postgresql.JSONB, nullable=False),
        sa.Column('rules', postgresql.JSONB, nullable=False),
        sa.Column('priority', sa.Integer, nullable=False, default=0),
        sa.Column('enabled', sa.Boolean, nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )
    
    # 创建索引
    op.create_index('idx_access_policies_enabled', 'access_policies', ['enabled'])
    op.create_index('idx_access_policies_priority', 'access_policies', ['priority'])
    
    # 3. 角色定义表
    op.create_table(
        'security_roles',
        sa.Column('role_id', sa.String(100), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('permissions', postgresql.ARRAY(sa.String(255)), nullable=False, default=[]),
        sa.Column('parent_roles', postgresql.ARRAY(sa.String(100)), nullable=False, default=[]),
        sa.Column('attributes', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )
    
    # 4. 主体表
    op.create_table(
        'security_subjects',
        sa.Column('subject_id', sa.String(255), primary_key=True),
        sa.Column('roles', postgresql.ARRAY(sa.String(100)), nullable=False, default=[]),
        sa.Column('attributes', postgresql.JSONB, nullable=True),
        sa.Column('active', sa.Boolean, nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )
    
    # 创建索引
    op.create_index('idx_security_subjects_active', 'security_subjects', ['active'])
    
    # 5. 安全事件表
    op.create_table(
        'security_events',
        sa.Column('event_id', sa.String(255), primary_key=True),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('timestamp', sa.BigInteger, nullable=False),
        sa.Column('source_agent_id', sa.String(255), nullable=False),
        sa.Column('target_resource', sa.String(255), nullable=True),
        sa.Column('action', sa.String(255), nullable=False),
        sa.Column('result', sa.String(50), nullable=False),
        sa.Column('details', postgresql.JSONB, nullable=True),
        sa.Column('threat_level', sa.String(20), nullable=False, default='low'),
        sa.Column('risk_score', sa.Numeric(3, 2), nullable=False, default=0.0),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )
    
    # 创建索引
    op.create_index('idx_security_events_timestamp', 'security_events', ['timestamp'])
    op.create_index('idx_security_events_agent', 'security_events', ['source_agent_id'])
    op.create_index('idx_security_events_type', 'security_events', ['event_type'])
    op.create_index('idx_security_events_threat_level', 'security_events', ['threat_level'])
    op.create_index('idx_security_events_result', 'security_events', ['result'])
    
    # 6. 安全告警表
    op.create_table(
        'security_alerts',
        sa.Column('alert_id', sa.String(255), primary_key=True),
        sa.Column('threat_pattern_id', sa.String(255), nullable=False),
        sa.Column('triggered_events', postgresql.ARRAY(sa.String(255)), nullable=False, default=[]),
        sa.Column('timestamp', sa.BigInteger, nullable=False),
        sa.Column('threat_level', sa.String(20), nullable=False),
        sa.Column('confidence_score', sa.Numeric(3, 2), nullable=False),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('recommended_actions', postgresql.ARRAY(sa.Text), nullable=False, default=[]),
        sa.Column('resolved', sa.Boolean, nullable=False, default=False),
        sa.Column('resolution_notes', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True)
    )
    
    # 创建索引
    op.create_index('idx_security_alerts_timestamp', 'security_alerts', ['timestamp'])
    op.create_index('idx_security_alerts_resolved', 'security_alerts', ['resolved'])
    op.create_index('idx_security_alerts_threat_level', 'security_alerts', ['threat_level'])
    
    # 7. 通信会话表
    op.create_table(
        'communication_sessions',
        sa.Column('session_id', sa.String(255), primary_key=True),
        sa.Column('participants', postgresql.ARRAY(sa.String(255)), nullable=False),
        sa.Column('key_version', sa.Integer, nullable=False, default=1),
        sa.Column('created_at', sa.BigInteger, nullable=False),
        sa.Column('expires_at', sa.BigInteger, nullable=False),
        sa.Column('forward_secure', sa.Boolean, nullable=False, default=True),
        sa.Column('status', sa.String(20), nullable=False, default='active')  # active, expired, closed
    )
    
    # 创建索引
    op.create_index('idx_communication_sessions_expires', 'communication_sessions', ['expires_at'])
    op.create_index('idx_communication_sessions_status', 'communication_sessions', ['status'])
    
    # 8. 访问日志表
    op.create_table(
        'access_logs',
        sa.Column('log_id', sa.String(255), primary_key=True),
        sa.Column('request_id', sa.String(255), nullable=False),
        sa.Column('subject_id', sa.String(255), nullable=False),
        sa.Column('resource_id', sa.String(255), nullable=False),
        sa.Column('action', sa.String(255), nullable=False),
        sa.Column('decision', sa.String(50), nullable=False),
        sa.Column('reason', sa.Text, nullable=True),
        sa.Column('timestamp', sa.BigInteger, nullable=False),
        sa.Column('evaluation_time_ms', sa.Float, nullable=False),
        sa.Column('context', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )
    
    # 创建索引
    op.create_index('idx_access_logs_timestamp', 'access_logs', ['timestamp'])
    op.create_index('idx_access_logs_subject', 'access_logs', ['subject_id'])
    op.create_index('idx_access_logs_resource', 'access_logs', ['resource_id'])
    op.create_index('idx_access_logs_decision', 'access_logs', ['decision'])
    
    # 9. 威胁模式表
    op.create_table(
        'threat_patterns',
        sa.Column('pattern_id', sa.String(255), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('event_types', postgresql.ARRAY(sa.String(50)), nullable=False),
        sa.Column('conditions', postgresql.JSONB, nullable=False),
        sa.Column('threat_level', sa.String(20), nullable=False),
        sa.Column('confidence_threshold', sa.Numeric(3, 2), nullable=False),
        sa.Column('enabled', sa.Boolean, nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )
    
    # 创建索引
    op.create_index('idx_threat_patterns_enabled', 'threat_patterns', ['enabled'])
    op.create_index('idx_threat_patterns_threat_level', 'threat_patterns', ['threat_level'])
    
    # 10. 密钥管理表
    op.create_table(
        'agent_keys',
        sa.Column('agent_id', sa.String(255), primary_key=True),
        sa.Column('public_key', sa.Text, nullable=False),
        sa.Column('key_algorithm', sa.String(50), nullable=False, default='RSA-4096'),
        sa.Column('key_fingerprint', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('revoked', sa.Boolean, nullable=False, default=False),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True)
    )
    
    # 创建索引
    op.create_index('idx_agent_keys_fingerprint', 'agent_keys', ['key_fingerprint'])
    op.create_index('idx_agent_keys_revoked', 'agent_keys', ['revoked'])
    
    # 11. 证书管理表
    op.create_table(
        'certificates',
        sa.Column('cert_id', sa.String(255), primary_key=True),
        sa.Column('serial_number', sa.String(255), nullable=False),
        sa.Column('subject', sa.String(500), nullable=False),
        sa.Column('issuer', sa.String(500), nullable=False),
        sa.Column('certificate_pem', sa.Text, nullable=False),
        sa.Column('not_before', sa.DateTime(timezone=True), nullable=False),
        sa.Column('not_after', sa.DateTime(timezone=True), nullable=False),
        sa.Column('revoked', sa.Boolean, nullable=False, default=False),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('revocation_reason', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )
    
    # 创建索引
    op.create_index('idx_certificates_serial', 'certificates', ['serial_number'])
    op.create_index('idx_certificates_subject', 'certificates', ['subject'])
    op.create_index('idx_certificates_revoked', 'certificates', ['revoked'])
    op.create_index('idx_certificates_not_after', 'certificates', ['not_after'])
    
    # 12. 审计配置表
    op.create_table(
        'audit_config',
        sa.Column('config_id', sa.String(255), primary_key=True),
        sa.Column('config_name', sa.String(255), nullable=False),
        sa.Column('config_value', postgresql.JSONB, nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )
    
    # 创建唯一索引
    op.create_unique_constraint('uq_audit_config_name', 'audit_config', ['config_name'])
    
    # 插入默认数据
    
    # 插入默认角色
    op.execute("""
        INSERT INTO security_roles (role_id, name, description, permissions, parent_roles) VALUES 
        ('admin', 'Administrator', 'System administrator with full access', ARRAY['*:*:*'], ARRAY[]::text[]),
        ('agent', 'Agent', 'AI agent with standard permissions', ARRAY['api_endpoint:read:*', 'api_endpoint:write:own_data/*', 'ai_model:invoke:*'], ARRAY[]::text[]),
        ('user', 'User', 'Regular user with limited access', ARRAY['api_endpoint:read:public/*', 'api_endpoint:write:user_data/*'], ARRAY[]::text[]),
        ('supervisor', 'Supervisor', 'Agent supervisor with monitoring capabilities', ARRAY['agent:monitor:*', 'agent:control:*', 'api_endpoint:read:*'], ARRAY['agent']::text[])
    """)
    
    # 插入默认策略
    op.execute("""
        INSERT INTO access_policies (policy_id, name, description, target, rules, priority) VALUES 
        ('admin_policy', 'Administrator Policy', 'Full access for administrators', 
         '{"subjects": ["admin"]}', '[{"condition": {}, "effect": "permit"}]', 100),
        ('basic_policy', 'Basic Access Policy', 'Basic access for authenticated agents', 
         '{"resource_type": "api_endpoint"}', '[{"condition": {"attribute": "subject.authenticated", "value": true, "operator": "equals"}, "effect": "permit"}]', 10),
        ('deny_all', 'Default Deny Policy', 'Deny all access by default', 
         '{}', '[{"condition": {}, "effect": "deny"}]', 0)
    """)
    
    # 插入默认威胁模式
    op.execute("""
        INSERT INTO threat_patterns (pattern_id, name, description, event_types, conditions, threat_level, confidence_threshold) VALUES 
        ('brute_force', '暴力破解攻击', '检测短时间内多次失败的认证尝试', 
         ARRAY['authentication']::text[], '{"time_window": 300, "min_failures": 5, "failure_rate_threshold": 0.8}', 'high', 0.9),
        ('privilege_escalation', '权限升级攻击', '检测异常的权限获取或使用行为', 
         ARRAY['authorization']::text[], '{"unusual_resource_access": true, "role_change_frequency_threshold": 3}', 'critical', 0.8),
        ('data_exfiltration', '数据泄露攻击', '检测异常的数据访问和传输行为', 
         ARRAY['data_access']::text[], '{"large_data_access": true, "unusual_time_access": true, "data_volume_threshold": 1000000}', 'critical', 0.7),
        ('lateral_movement', '横向移动攻击', '检测在网络中的异常移动行为', 
         ARRAY['network_activity']::text[], '{"unusual_network_connections": true, "connection_frequency_threshold": 10, "time_window": 600}', 'high', 0.8)
    """)
    
    # 插入默认审计配置
    op.execute("""
        INSERT INTO audit_config (config_id, config_name, config_value, description) VALUES 
        ('detection_window', 'Detection Time Window', '300', '威胁检测时间窗口（秒）'),
        ('anomaly_threshold', 'Anomaly Threshold', '0.8', '异常检测阈值'),
        ('max_failed_attempts', 'Max Failed Attempts', '5', '最大失败尝试次数'),
        ('session_timeout', 'Session Timeout', '3600', '会话超时时间（秒）'),
        ('min_trust_score', 'Minimum Trust Score', '0.6', '最低信任分数')
    """)


def downgrade():
    """删除分布式安全框架数据库表"""
    
    # 按相反顺序删除表
    op.drop_table('audit_config')
    op.drop_table('certificates')
    op.drop_table('agent_keys')
    op.drop_table('threat_patterns')
    op.drop_table('access_logs')
    op.drop_table('communication_sessions')
    op.drop_table('security_alerts')
    op.drop_table('security_events')
    op.drop_table('security_subjects')
    op.drop_table('security_roles')
    op.drop_table('access_policies')
    op.drop_table('agent_identities')