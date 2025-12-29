"""Create experiment tables for A/B testing platform

Revision ID: 002
Revises: 001_pgvector_0_8_upgrade
Create Date: 2025-01-20 10:00:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '002'
down_revision = '001_pgvector_0_8_upgrade'
branch_labels = None
depends_on = None

def upgrade():
    """Create experiment tables and indexes"""
    
    # Create experiments table
    op.create_table(
        'experiments',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('hypothesis', sa.Text(), nullable=False),
        sa.Column('owner', sa.String(255), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='draft'),
        sa.Column('start_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('end_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('success_metrics', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('guardrail_metrics', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('minimum_sample_size', sa.Integer(), nullable=False),
        sa.Column('significance_level', sa.Float(), nullable=False, server_default='0.05'),
        sa.Column('power', sa.Float(), nullable=False, server_default='0.8'),
        sa.Column('layers', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('targeting_rules', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint("status IN ('draft', 'running', 'paused', 'completed', 'terminated')", name='check_experiment_status'),
        sa.CheckConstraint('minimum_sample_size >= 100', name='check_minimum_sample_size'),
        sa.CheckConstraint('significance_level >= 0.01 AND significance_level <= 0.1', name='check_significance_level'),
        sa.CheckConstraint('power >= 0.5 AND power <= 0.99', name='check_power'),
        sa.CheckConstraint('end_date IS NULL OR end_date > start_date', name='check_end_date_after_start'),
    )
    
    # Create indexes for experiments table
    op.create_index('ix_experiments_name', 'experiments', ['name'])
    op.create_index('ix_experiments_owner', 'experiments', ['owner'])
    op.create_index('ix_experiments_status', 'experiments', ['status'])
    op.create_index('ix_experiments_status_start_date', 'experiments', ['status', 'start_date'])
    op.create_index('ix_experiments_owner_status', 'experiments', ['owner', 'status'])
    
    # Create experiment_variants table
    op.create_table(
        'experiment_variants',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('experiment_id', sa.String(), nullable=False),
        sa.Column('variant_id', sa.String(100), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('config', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('is_control', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('traffic_percentage', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('experiment_id', 'variant_id', name='uq_experiment_variant'),
        sa.CheckConstraint('traffic_percentage >= 0 AND traffic_percentage <= 100', name='check_traffic_percentage'),
    )
    
    # Create indexes for experiment_variants table
    op.create_index('ix_variants_experiment_id', 'experiment_variants', ['experiment_id'])
    op.create_index('ix_variants_experiment_variant', 'experiment_variants', ['experiment_id', 'variant_id'])
    
    # Create experiment_assignments table
    op.create_table(
        'experiment_assignments',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('experiment_id', sa.String(), nullable=False),
        sa.Column('variant_id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('context', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('is_eligible', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('assignment_reason', sa.String(100), nullable=False, server_default='traffic_split'),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('experiment_id', 'user_id', name='uq_experiment_user_assignment'),
    )
    
    # Create indexes for experiment_assignments table
    op.create_index('ix_assignments_experiment_id', 'experiment_assignments', ['experiment_id'])
    op.create_index('ix_assignments_variant_id', 'experiment_assignments', ['variant_id'])
    op.create_index('ix_assignments_user_id', 'experiment_assignments', ['user_id'])
    op.create_index('ix_assignments_timestamp', 'experiment_assignments', ['timestamp'])
    op.create_index('ix_assignments_user_experiment', 'experiment_assignments', ['user_id', 'experiment_id'])
    op.create_index('ix_assignments_experiment_variant', 'experiment_assignments', ['experiment_id', 'variant_id'])
    
    # Create experiment_events table
    op.create_table(
        'experiment_events',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('experiment_id', sa.String(), nullable=False),
        sa.Column('assignment_id', sa.String(), nullable=False),
        sa.Column('variant_id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('event_type', sa.String(100), nullable=False),
        sa.Column('event_value', sa.Float(), nullable=False, server_default='1.0'),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('processed', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['assignment_id'], ['experiment_assignments.id'], ondelete='CASCADE'),
    )
    
    # Create indexes for experiment_events table
    op.create_index('ix_events_experiment_id', 'experiment_events', ['experiment_id'])
    op.create_index('ix_events_assignment_id', 'experiment_events', ['assignment_id'])
    op.create_index('ix_events_variant_id', 'experiment_events', ['variant_id'])
    op.create_index('ix_events_user_id', 'experiment_events', ['user_id'])
    op.create_index('ix_events_event_type', 'experiment_events', ['event_type'])
    op.create_index('ix_events_timestamp', 'experiment_events', ['timestamp'])
    op.create_index('ix_events_processed', 'experiment_events', ['processed'])
    op.create_index('ix_events_experiment_type_timestamp', 'experiment_events', ['experiment_id', 'event_type', 'timestamp'])
    op.create_index('ix_events_user_experiment_timestamp', 'experiment_events', ['user_id', 'experiment_id', 'timestamp'])
    op.create_index('ix_events_variant_type_timestamp', 'experiment_events', ['variant_id', 'event_type', 'timestamp'])
    
    # Create experiment_metric_results table
    op.create_table(
        'experiment_metric_results',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('experiment_id', sa.String(), nullable=False),
        sa.Column('metric_name', sa.String(255), nullable=False),
        sa.Column('variant_results', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('statistical_test', sa.String(100), nullable=False),
        sa.Column('p_value', sa.Float(), nullable=False),
        sa.Column('is_significant', sa.Boolean(), nullable=False),
        sa.Column('effect_size', sa.Float(), nullable=False),
        sa.Column('confidence_interval_lower', sa.Float(), nullable=False),
        sa.Column('confidence_interval_upper', sa.Float(), nullable=False),
        sa.Column('sample_size', sa.Integer(), nullable=False),
        sa.Column('statistical_power', sa.Float(), nullable=False),
        sa.Column('computed_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('data_window_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('data_window_end', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('experiment_id', 'metric_name', 'computed_at', name='uq_experiment_metric_computed'),
        sa.CheckConstraint('p_value >= 0 AND p_value <= 1', name='check_p_value'),
        sa.CheckConstraint('statistical_power >= 0 AND statistical_power <= 1', name='check_statistical_power'),
        sa.CheckConstraint('sample_size >= 0', name='check_sample_size'),
        sa.CheckConstraint('confidence_interval_lower <= confidence_interval_upper', name='check_confidence_interval'),
    )
    
    # Create indexes for experiment_metric_results table
    op.create_index('ix_metric_results_experiment_id', 'experiment_metric_results', ['experiment_id'])
    op.create_index('ix_metric_results_metric_name', 'experiment_metric_results', ['metric_name'])
    op.create_index('ix_metric_results_computed_at', 'experiment_metric_results', ['computed_at'])
    op.create_index('ix_metric_results_experiment_metric', 'experiment_metric_results', ['experiment_id', 'metric_name'])
    
    # Create experiment_layer_conflicts table
    op.create_table(
        'experiment_layer_conflicts',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('layer', sa.String(100), nullable=False),
        sa.Column('experiment_id_1', sa.String(), nullable=False),
        sa.Column('experiment_id_2', sa.String(), nullable=False),
        sa.Column('conflict_type', sa.String(50), nullable=False),
        sa.Column('detected_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('resolution_method', sa.String(100), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['experiment_id_1'], ['experiments.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['experiment_id_2'], ['experiments.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('layer', 'experiment_id_1', 'experiment_id_2', name='uq_layer_experiment_conflict'),
    )
    
    # Create indexes for experiment_layer_conflicts table
    op.create_index('ix_layer_conflicts_layer', 'experiment_layer_conflicts', ['layer'])
    op.create_index('ix_layer_conflicts_detected', 'experiment_layer_conflicts', ['detected_at'])

def downgrade():
    """Drop experiment tables"""
    
    # Drop tables in reverse order to handle foreign key constraints
    op.drop_table('experiment_layer_conflicts')
    op.drop_table('experiment_metric_results')
    op.drop_table('experiment_events')
    op.drop_table('experiment_assignments')
    op.drop_table('experiment_variants')
    op.drop_table('experiments')
