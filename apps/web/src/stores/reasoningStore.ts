import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { reasoningService } from '../services/reasoningService';

interface ReasoningChain {
  id: string;
  strategy: string;
  problem: string;
  context?: string;
  conclusion?: string;
  confidence_score?: number;
  created_at: string;
  completed_at?: string;
  steps: Array<{
    id: string;
    step_number: number;
    step_type: string;
    content: string;
    reasoning: string;
    confidence: number;
    duration_ms?: number;
  }>;
  branches?: Array<{
    id: string;
    parent_step_id: string;
    reason: string;
    created_at: string;
  }>;
}

interface ReasoningRequest {
  problem: string;
  strategy: 'ZERO_SHOT' | 'FEW_SHOT' | 'AUTO_COT';
  context?: string;
  max_steps: number;
  stream: boolean;
  enable_branching: boolean;
  examples?: Array<{
    problem: string;
    reasoning: string;
    answer: string;
  }>;
}

interface ReasoningStreamChunk {
  chain_id: string;
  step_number: number;
  step_type: string;
  content: string;
  reasoning: string;
  confidence: number;
  is_final: boolean;
}

interface ReasoningValidation {
  step_id: string;
  is_valid: boolean;
  consistency_score: number;
  issues: string[];
  suggestions: string[];
}

interface RecoveryStats {
  total_failures: number;
  recovery_attempts: number;
  recovery_success_rate: number;
  strategy_effectiveness: Record<string, number>;
}

interface ReasoningStats {
  total_chains: number;
  completed_chains: number;
  avg_confidence: number;
  avg_steps: number;
  strategy_distribution: Record<string, number>;
  quality_metrics: {
    high_quality: number;
    medium_quality: number;
    low_quality: number;
  };
}

interface ReasoningStore {
  // 状态
  currentChain: ReasoningChain | null;
  reasoningHistory: ReasoningChain[];
  streamingSteps: ReasoningStreamChunk[];
  validationResults: ReasoningValidation | null;
  recoveryStats: RecoveryStats | null;
  reasoningStats: ReasoningStats | null;
  isExecuting: boolean;
  isLoading: boolean;
  error: string | null;

  // 动作
  executeReasoning: (request: ReasoningRequest) => Promise<ReasoningChain>;
  streamReasoning: (request: ReasoningRequest) => Promise<void>;
  validateChain: (chainId: string, stepNumber?: number) => Promise<ReasoningValidation>;
  recoverChain: (chainId: string, strategy?: string) => Promise<boolean>;
  createBranch: (chainId: string, parentStepNumber: number, reason: string) => Promise<string>;
  getReasoningChain: (chainId: string) => Promise<ReasoningChain>;
  getReasoningHistory: (limit?: number, offset?: number) => Promise<void>;
  deleteReasoningChain: (chainId: string) => Promise<void>;
  getReasoningStats: () => Promise<void>;
  setCurrentChain: (chain: ReasoningChain | null) => void;
  clearStreamingSteps: () => void;
  addStreamingStep: (step: ReasoningStreamChunk) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

export const useReasoningStore = create<ReasoningStore>()(
  devtools(
    (set, get) => ({
      // 初始状态
      currentChain: null,
      reasoningHistory: [],
      streamingSteps: [],
      validationResults: null,
      recoveryStats: null,
      reasoningStats: null,
      isExecuting: false,
      isLoading: false,
      error: null,

      // 执行推理
      executeReasoning: async (request: ReasoningRequest) => {
        set({ isExecuting: true, error: null });
        
        try {
          const chain = await reasoningService.executeReasoning(request);
          set({ 
            currentChain: chain, 
            isExecuting: false 
          });
          return chain;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : '推理执行失败';
          set({ 
            error: errorMessage, 
            isExecuting: false 
          });
          throw error;
        }
      },

      // 流式推理
      streamReasoning: async (request: ReasoningRequest) => {
        set({ isExecuting: true, error: null, streamingSteps: [] });
        
        try {
          await reasoningService.streamReasoning(request, (chunk) => {
            get().addStreamingStep(chunk);
            
            // 如果是最终步骤，更新当前链
            if (chunk.is_final) {
              // 这里可以构建完整的推理链
              const chain: ReasoningChain = {
                id: chunk.chain_id,
                strategy: request.strategy,
                problem: request.problem,
                context: request.context,
                created_at: new Date().toISOString(),
                steps: get().streamingSteps.map(s => ({
                  id: `${s.chain_id}-${s.step_number}`,
                  step_number: s.step_number,
                  step_type: s.step_type,
                  content: s.content,
                  reasoning: s.reasoning,
                  confidence: s.confidence
                }))
              };
              set({ currentChain: chain });
            }
          });
          
          set({ isExecuting: false });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : '流式推理失败';
          set({ 
            error: errorMessage, 
            isExecuting: false 
          });
          throw error;
        }
      },

      // 验证推理链
      validateChain: async (chainId: string, stepNumber?: number) => {
        set({ isLoading: true, error: null });
        
        try {
          const validation = await reasoningService.validateChain(chainId, stepNumber);
          set({ 
            validationResults: validation, 
            isLoading: false 
          });
          return validation;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : '验证失败';
          set({ 
            error: errorMessage, 
            isLoading: false 
          });
          throw error;
        }
      },

      // 恢复推理链
      recoverChain: async (chainId: string, strategy?: string) => {
        set({ isLoading: true, error: null });
        
        try {
          const success = await reasoningService.recoverChain(chainId, strategy);
          set({ isLoading: false });
          
          if (success) {
            // 重新获取推理链
            const chain = await reasoningService.getReasoningChain(chainId);
            set({ currentChain: chain });
          }
          
          return success;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : '恢复失败';
          set({ 
            error: errorMessage, 
            isLoading: false 
          });
          throw error;
        }
      },

      // 创建分支
      createBranch: async (chainId: string, parentStepNumber: number, reason: string) => {
        set({ isLoading: true, error: null });
        
        try {
          const branchId = await reasoningService.createBranch(chainId, parentStepNumber, reason);
          set({ isLoading: false });
          
          // 重新获取推理链以显示分支
          const chain = await reasoningService.getReasoningChain(chainId);
          set({ currentChain: chain });
          
          return branchId;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : '创建分支失败';
          set({ 
            error: errorMessage, 
            isLoading: false 
          });
          throw error;
        }
      },

      // 获取推理链
      getReasoningChain: async (chainId: string) => {
        set({ isLoading: true, error: null });
        
        try {
          const chain = await reasoningService.getReasoningChain(chainId);
          set({ 
            currentChain: chain, 
            isLoading: false 
          });
          return chain;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : '获取推理链失败';
          set({ 
            error: errorMessage, 
            isLoading: false 
          });
          throw error;
        }
      },

      // 获取推理历史
      getReasoningHistory: async (limit = 20, offset = 0) => {
        set({ isLoading: true, error: null });
        
        try {
          const history = await reasoningService.getReasoningHistory(limit, offset);
          set({ 
            reasoningHistory: history, 
            isLoading: false 
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : '获取历史失败';
          set({ 
            error: errorMessage, 
            isLoading: false 
          });
        }
      },

      // 删除推理链
      deleteReasoningChain: async (chainId: string) => {
        set({ isLoading: true, error: null });
        
        try {
          await reasoningService.deleteReasoningChain(chainId);
          
          // 从历史中移除
          set(state => ({
            reasoningHistory: state.reasoningHistory.filter(chain => chain.id !== chainId),
            isLoading: false,
            // 如果删除的是当前链，清空当前链
            currentChain: state.currentChain?.id === chainId ? null : state.currentChain
          }));
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : '删除失败';
          set({ 
            error: errorMessage, 
            isLoading: false 
          });
          throw error;
        }
      },

      // 获取统计信息
      getReasoningStats: async () => {
        set({ isLoading: true, error: null });
        
        try {
          const stats = await reasoningService.getReasoningStats();
          set({ 
            reasoningStats: stats, 
            isLoading: false 
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : '获取统计失败';
          set({ 
            error: errorMessage, 
            isLoading: false 
          });
        }
      },

      // 设置当前推理链
      setCurrentChain: (chain: ReasoningChain | null) => {
        set({ currentChain: chain });
      },

      // 清空流式步骤
      clearStreamingSteps: () => {
        set({ streamingSteps: [] });
      },

      // 添加流式步骤
      addStreamingStep: (step: ReasoningStreamChunk) => {
        set(state => ({
          streamingSteps: [...state.streamingSteps, step]
        }));
      },

      // 设置错误
      setError: (error: string | null) => {
        set({ error });
      },

      // 重置状态
      reset: () => {
        set({
          currentChain: null,
          streamingSteps: [],
          validationResults: null,
          recoveryStats: null,
          isExecuting: false,
          error: null
        });
      }
    }),
    {
      name: 'reasoning-store',
      partialize: (state) => ({
        // 只持久化历史记录
        reasoningHistory: state.reasoningHistory
      })
    }
  )
);