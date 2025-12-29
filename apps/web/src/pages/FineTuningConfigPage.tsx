import React, { useEffect, useMemo, useState } from 'react';
import { Card, Table, Tag, Form, Input, Select, Button, Alert, Space, message } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { fineTuningService, TrainingJobRequest } from '../services/fineTuningService';

import { logger } from '../utils/logger'
type TemplateRecord = {
  name: string;
  training_mode?: string;
  learning_rate?: number;
  num_train_epochs?: number;
  per_device_train_batch_size?: number;
  gradient_accumulation_steps?: number;
  lora_config?: any;
  quantization_config?: any;
};

const FineTuningConfigPage: React.FC = () => {
  const [templates, setTemplates] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState(false);
  const [validating, setValidating] = useState(false);
  const [validationResult, setValidationResult] = useState<{ valid: boolean; errors: string[] } | null>(null);
  const [form] = Form.useForm();

  const loadTemplates = async () => {
    try {
      setLoading(true);
      const data = await fineTuningService.getConfigTemplates();
      setTemplates(data.templates || {});
    } catch (e) {
      logger.error('加载配置模板失败:', e);
      message.error('加载配置模板失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadTemplates();
  }, []);

  const templateRows = useMemo<TemplateRecord[]>(
    () =>
      Object.entries(templates).map(([name, tpl]) => ({
        name,
        training_mode: tpl?.training_mode,
        learning_rate: tpl?.learning_rate,
        num_train_epochs: tpl?.num_train_epochs,
        per_device_train_batch_size: tpl?.per_device_train_batch_size,
        gradient_accumulation_steps: tpl?.gradient_accumulation_steps,
        lora_config: tpl?.lora_config,
        quantization_config: tpl?.quantization_config,
      })),
    [templates]
  );

  const columns: ColumnsType<TemplateRecord> = [
    { title: '模板名称', dataIndex: 'name', key: 'name' },
    {
      title: '训练模式',
      dataIndex: 'training_mode',
      key: 'training_mode',
      render: (value: string) => <Tag>{value || '-'}</Tag>,
    },
    {
      title: '学习率',
      dataIndex: 'learning_rate',
      key: 'learning_rate',
      render: (value: number) => (value ? value.toString() : '-'),
    },
    {
      title: '轮次',
      dataIndex: 'num_train_epochs',
      key: 'num_train_epochs',
      render: (value: number) => (value ?? '-'),
    },
    {
      title: '批次大小',
      dataIndex: 'per_device_train_batch_size',
      key: 'per_device_train_batch_size',
      render: (value: number) => (value ?? '-'),
    },
    {
      title: '量化',
      dataIndex: 'quantization_config',
      key: 'quantization_config',
      render: (value: any) => (value?.quantization_type ? <Tag>{value.quantization_type}</Tag> : '-'),
    },
  ];

  const handleValidate = async () => {
    try {
      const values = await form.validateFields();
      const selected = templates[values.template_name] || {};
      const payload: TrainingJobRequest = {
        job_name: values.job_name,
        model_name: values.model_name,
        dataset_path: values.dataset_path,
        output_dir: values.output_dir || undefined,
        training_mode: selected.training_mode || 'lora',
        learning_rate: selected.learning_rate ?? 0.0002,
        num_train_epochs: selected.num_train_epochs ?? 3,
        per_device_train_batch_size: selected.per_device_train_batch_size ?? 4,
        gradient_accumulation_steps: selected.gradient_accumulation_steps ?? 4,
        warmup_steps: selected.warmup_steps ?? 100,
        max_seq_length: selected.max_seq_length ?? 2048,
        lora_config: selected.lora_config,
        quantization_config: selected.quantization_config,
        use_distributed: selected.use_distributed ?? false,
        use_deepspeed: selected.use_deepspeed ?? false,
        use_flash_attention: selected.use_flash_attention ?? true,
        use_gradient_checkpointing: selected.use_gradient_checkpointing ?? true,
        fp16: selected.fp16 ?? false,
        bf16: selected.bf16 ?? true,
      };
      setValidating(true);
      const result = await fineTuningService.validateTrainingConfig(payload);
      setValidationResult(result);
      message.success(result.valid ? '配置验证通过' : '配置验证未通过');
    } catch (e: any) {
      if (e?.errorFields) return;
      logger.error('验证配置失败:', e);
      message.error('验证配置失败');
    } finally {
      setValidating(false);
    }
  };

  return (
    <div style={{ padding: '24px' }}>
      <Card title="配置模板" style={{ marginBottom: 16 }}>
        <Table columns={columns} dataSource={templateRows} rowKey="name" loading={loading} pagination={false} />
      </Card>

      <Card title="配置校验">
        <Form form={form} layout="vertical">
          <Form.Item
            name="template_name"
            label="选择模板"
            rules={[{ required: true, message: '请选择模板' }]}
          >
            <Select placeholder="请选择模板">
              {Object.keys(templates).map(name => (
                <Select.Option key={name} value={name}>
                  {name}
                </Select.Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item name="job_name" label="任务名称" rules={[{ required: true, message: '请输入任务名称' }]}>
            <Input placeholder="例如 lora_small_job" />
          </Form.Item>
          <Form.Item name="model_name" label="模型名称" rules={[{ required: true, message: '请输入模型名称' }]}>
            <Input placeholder="例如 hf-internal-testing/tiny-random-LlamaForCausalLM" />
          </Form.Item>
          <Form.Item name="dataset_path" label="数据集路径" rules={[{ required: true, message: '请输入数据集路径' }]}>
            <Input placeholder="例如 data/train.jsonl" />
          </Form.Item>
          <Form.Item name="output_dir" label="输出目录">
            <Input placeholder="例如 ./fine_tuned_models" />
          </Form.Item>
          <Space>
            <Button type="primary" onClick={handleValidate} loading={validating}>
              校验配置
            </Button>
            <Button onClick={loadTemplates}>刷新模板</Button>
          </Space>
        </Form>

        {validationResult && (
          <Alert
            style={{ marginTop: 16 }}
            type={validationResult.valid ? 'success' : 'error'}
            message={validationResult.valid ? '配置验证通过' : '配置验证未通过'}
            description={
              validationResult.valid
                ? '可以直接创建微调任务'
                : validationResult.errors?.length
                  ? validationResult.errors.join('；')
                  : '请检查输入参数'
            }
            showIcon
          />
        )}
      </Card>
    </div>
  );
};

export default FineTuningConfigPage;
