from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
import json
import yaml
from enum import Enum
import logging

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """基准测试管理器配置"""
    config_path: str = "config/benchmarks.yaml"
    cache_dir: str = "cache/benchmarks"
    max_concurrent_downloads: int = 3
    download_timeout_seconds: int = 300
    enable_caching: bool = True
    auto_update_benchmarks: bool = False
    custom_benchmark_dir: Optional[str] = None
    allowed_benchmark_types: Optional[Set[str]] = None
    max_benchmark_size_gb: float = 10.0
    validation_enabled: bool = True

class BenchmarkType(Enum):
    """基准测试类型"""
    LANGUAGE_UNDERSTANDING = "language_understanding"
    TEXT_GENERATION = "text_generation"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    KNOWLEDGE = "knowledge"
    SAFETY = "safety"
    MULTIMODAL = "multimodal"

class DifficultyLevel(Enum):
    """难度级别"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"

@dataclass
class BenchmarkInfo:
    """基准测试信息"""
    name: str
    display_name: str
    description: str
    benchmark_type: BenchmarkType
    difficulty: DifficultyLevel
    tasks: List[str]
    languages: List[str] = field(default_factory=lambda: ["en"])
    num_samples: Optional[int] = None
    num_fewshot: int = 0
    metrics: List[str] = field(default_factory=list)
    paper_url: Optional[str] = None
    homepage: Optional[str] = None
    citation: Optional[str] = None
    requires_special_setup: bool = False
    estimated_runtime_minutes: Optional[int] = None
    memory_requirements_gb: Optional[float] = None

@dataclass 
class BenchmarkSuite:
    """基准测试套件"""
    name: str
    description: str
    benchmarks: List[BenchmarkInfo] = field(default_factory=list)
    benchmark_names: List[str] = field(default_factory=list)
    display_name: Optional[str] = None
    total_tasks: int = 0
    estimated_runtime_hours: float = 0.0
    created_at: datetime = field(default_factory=utc_factory)
    
    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name
        if self.benchmarks:
            self.total_tasks = len(self.benchmarks)
            self.estimated_runtime_hours = sum(
                b.estimated_runtime_minutes or 30 for b in self.benchmarks
            ) / 60.0
            if not self.benchmark_names:
                self.benchmark_names = [b.name for b in self.benchmarks]
        elif self.benchmark_names:
            self.total_tasks = len(self.benchmark_names)

class BenchmarkManager:
    """基准测试管理器"""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.benchmarks: Dict[str, BenchmarkInfo] = {}
        self.suites: Dict[str, BenchmarkSuite] = {}
        self.config_path = self.config.config_path
        self._initialize_standard_benchmarks()
        self._load_custom_benchmarks()
    
    def _initialize_standard_benchmarks(self):
        """初始化标准基准测试"""
        
        # GLUE基准测试套件
        glue_benchmarks = [
            BenchmarkInfo(
                name="cola",
                display_name="CoLA",
                description="Corpus of Linguistic Acceptability",
                benchmark_type=BenchmarkType.LANGUAGE_UNDERSTANDING,
                difficulty=DifficultyLevel.MEDIUM,
                tasks=["cola"],
                metrics=["accuracy", "f1"],
                num_samples=1043,
                paper_url="https://nyu-mll.github.io/CoLA/",
                estimated_runtime_minutes=5
            ),
            BenchmarkInfo(
                name="sst2",
                display_name="SST-2",
                description="Stanford Sentiment Treebank",
                benchmark_type=BenchmarkType.LANGUAGE_UNDERSTANDING,
                difficulty=DifficultyLevel.EASY,
                tasks=["sst2"],
                metrics=["accuracy"],
                num_samples=1821,
                estimated_runtime_minutes=8
            ),
            BenchmarkInfo(
                name="mrpc",
                display_name="MRPC",
                description="Microsoft Research Paraphrase Corpus",
                benchmark_type=BenchmarkType.LANGUAGE_UNDERSTANDING,
                difficulty=DifficultyLevel.MEDIUM,
                tasks=["mrpc"],
                metrics=["accuracy", "f1"],
                num_samples=408,
                estimated_runtime_minutes=5
            ),
            BenchmarkInfo(
                name="qqp",
                display_name="QQP",
                description="Quora Question Pairs",
                benchmark_type=BenchmarkType.LANGUAGE_UNDERSTANDING,
                difficulty=DifficultyLevel.MEDIUM,
                tasks=["qqp"],
                metrics=["accuracy", "f1"],
                num_samples=40430,
                estimated_runtime_minutes=45
            ),
            BenchmarkInfo(
                name="mnli",
                display_name="MNLI",
                description="Multi-Genre Natural Language Inference",
                benchmark_type=BenchmarkType.LANGUAGE_UNDERSTANDING,
                difficulty=DifficultyLevel.HARD,
                tasks=["mnli"],
                metrics=["accuracy"],
                num_samples=9815,
                estimated_runtime_minutes=25
            ),
        ]
        
        # SuperGLUE基准测试
        superglue_benchmarks = [
            BenchmarkInfo(
                name="boolq",
                display_name="BoolQ",
                description="Boolean Questions",
                benchmark_type=BenchmarkType.QUESTION_ANSWERING,
                difficulty=DifficultyLevel.MEDIUM,
                tasks=["boolq"],
                metrics=["accuracy"],
                num_samples=3270,
                estimated_runtime_minutes=15
            ),
            BenchmarkInfo(
                name="rte",
                display_name="RTE",
                description="Recognizing Textual Entailment",
                benchmark_type=BenchmarkType.LANGUAGE_UNDERSTANDING,
                difficulty=DifficultyLevel.HARD,
                tasks=["rte"],
                metrics=["accuracy"],
                num_samples=277,
                estimated_runtime_minutes=5
            ),
            BenchmarkInfo(
                name="wic",
                display_name="WiC",
                description="Words in Context",
                benchmark_type=BenchmarkType.LANGUAGE_UNDERSTANDING,
                difficulty=DifficultyLevel.HARD,
                tasks=["wic"],
                metrics=["accuracy"],
                num_samples=638,
                estimated_runtime_minutes=8
            ),
        ]
        
        # MMLU基准测试
        mmlu_benchmarks = [
            BenchmarkInfo(
                name="mmlu",
                display_name="MMLU",
                description="Massive Multitask Language Understanding",
                benchmark_type=BenchmarkType.KNOWLEDGE,
                difficulty=DifficultyLevel.HARD,
                tasks=["mmlu"],
                metrics=["accuracy"],
                num_samples=14042,
                paper_url="https://arxiv.org/abs/2009.03300",
                estimated_runtime_minutes=60,
                memory_requirements_gb=8.0
            )
        ]
        
        # 代码生成基准测试
        code_benchmarks = [
            BenchmarkInfo(
                name="humaneval",
                display_name="HumanEval",
                description="Evaluating Large Language Models Trained on Code",
                benchmark_type=BenchmarkType.CODE_GENERATION,
                difficulty=DifficultyLevel.HARD,
                tasks=["humaneval"],
                metrics=["pass@1", "pass@10", "pass@100"],
                num_samples=164,
                paper_url="https://arxiv.org/abs/2107.03374",
                estimated_runtime_minutes=30
            ),
            BenchmarkInfo(
                name="mbpp",
                display_name="MBPP",
                description="Mostly Basic Programming Problems",
                benchmark_type=BenchmarkType.CODE_GENERATION,
                difficulty=DifficultyLevel.MEDIUM,
                tasks=["mbpp"],
                metrics=["pass@1", "pass@10"],
                num_samples=974,
                estimated_runtime_minutes=45
            )
        ]
        
        # 推理基准测试
        reasoning_benchmarks = [
            BenchmarkInfo(
                name="hellaswag",
                display_name="HellaSwag",
                description="Can a Machine Really Finish Your Sentence?",
                benchmark_type=BenchmarkType.REASONING,
                difficulty=DifficultyLevel.HARD,
                tasks=["hellaswag"],
                metrics=["accuracy"],
                num_samples=10042,
                paper_url="https://arxiv.org/abs/1905.07830",
                estimated_runtime_minutes=25
            ),
            BenchmarkInfo(
                name="arc_easy",
                display_name="ARC-Easy",
                description="AI2 Reasoning Challenge (Easy Set)",
                benchmark_type=BenchmarkType.REASONING,
                difficulty=DifficultyLevel.EASY,
                tasks=["arc_easy"],
                metrics=["accuracy"],
                num_samples=2376,
                estimated_runtime_minutes=12
            ),
            BenchmarkInfo(
                name="arc_challenge",
                display_name="ARC-Challenge",
                description="AI2 Reasoning Challenge (Challenge Set)",
                benchmark_type=BenchmarkType.REASONING,
                difficulty=DifficultyLevel.HARD,
                tasks=["arc_challenge"],
                metrics=["accuracy"],
                num_samples=1172,
                estimated_runtime_minutes=15
            )
        ]
        
        # 注册所有基准测试
        all_benchmarks = (
            glue_benchmarks + superglue_benchmarks + mmlu_benchmarks + 
            code_benchmarks + reasoning_benchmarks
        )
        
        for benchmark in all_benchmarks:
            self.benchmarks[benchmark.name] = benchmark
        
        # 创建基准测试套件
        self.suites = {
            "glue": BenchmarkSuite(
                name="glue",
                description="General Language Understanding Evaluation",
                benchmarks=glue_benchmarks
            ),
            "superglue": BenchmarkSuite(
                name="superglue", 
                description="Super General Language Understanding Evaluation",
                benchmarks=superglue_benchmarks
            ),
            "code": BenchmarkSuite(
                name="code",
                description="Code Generation and Understanding",
                benchmarks=code_benchmarks
            ),
            "reasoning": BenchmarkSuite(
                name="reasoning",
                description="Common Sense and Logical Reasoning",
                benchmarks=reasoning_benchmarks
            ),
            "comprehensive": BenchmarkSuite(
                name="comprehensive",
                description="Comprehensive Model Evaluation Suite",
                benchmarks=all_benchmarks[:10]  # 选择前10个作为综合测试
            )
        }
    
    def _load_custom_benchmarks(self):
        """加载自定义基准测试配置"""
        config_file = Path(self.config_path)
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    custom_config = yaml.safe_load(f)
                    
                for bench_config in custom_config.get("benchmarks", []):
                    benchmark = BenchmarkInfo(**bench_config)
                    self.benchmarks[benchmark.name] = benchmark
                    logger.info(f"Loaded custom benchmark: {benchmark.name}")
                    
            except Exception as e:
                logger.warning(f"Failed to load custom benchmarks: {e}")
    
    def get_benchmark(self, name: str) -> Optional[BenchmarkInfo]:
        """获取基准测试信息"""
        return self.benchmarks.get(name)
    
    def get_benchmarks_by_type(self, benchmark_type: BenchmarkType) -> List[BenchmarkInfo]:
        """根据类型获取基准测试"""
        return [
            benchmark for benchmark in self.benchmarks.values()
            if benchmark.benchmark_type == benchmark_type
        ]
    
    def get_benchmarks_by_difficulty(self, difficulty: DifficultyLevel) -> List[BenchmarkInfo]:
        """根据难度获取基准测试"""
        return [
            benchmark for benchmark in self.benchmarks.values()
            if benchmark.difficulty == difficulty
        ]
    
    def get_suite(self, suite_name: str) -> Optional[BenchmarkSuite]:
        """获取基准测试套件"""
        return self.suites.get(suite_name)
    
    def list_benchmarks(self, 
                       benchmark_type: Optional[BenchmarkType] = None,
                       difficulty: Optional[DifficultyLevel] = None,
                       language: Optional[str] = None) -> List[BenchmarkInfo]:
        """列出基准测试"""
        benchmarks = list(self.benchmarks.values())
        
        if benchmark_type:
            benchmarks = [b for b in benchmarks if b.benchmark_type == benchmark_type]
        
        if difficulty:
            benchmarks = [b for b in benchmarks if b.difficulty == difficulty]
            
        if language:
            benchmarks = [b for b in benchmarks if language in b.languages]
            
        return sorted(benchmarks, key=lambda x: x.name)
    
    def list_suites(self) -> List[BenchmarkSuite]:
        """列出所有套件"""
        return list(self.suites.values())
    
    def create_custom_suite(self, 
                           name: str, 
                           description: str, 
                           benchmark_names: List[str]) -> BenchmarkSuite:
        """创建自定义套件"""
        benchmarks = []
        for bench_name in benchmark_names:
            benchmark = self.get_benchmark(bench_name)
            if benchmark:
                benchmarks.append(benchmark)
            else:
                logger.warning(f"Benchmark not found: {bench_name}")
        
        suite = BenchmarkSuite(
            name=name,
            description=description,
            benchmarks=benchmarks
        )
        
        self.suites[name] = suite
        return suite
    
    def get_benchmark_requirements(self, benchmark_names: List[str]) -> Dict[str, Any]:
        """获取基准测试资源需求"""
        total_samples = 0
        max_memory = 0.0
        total_time = 0
        required_languages = set()
        
        for name in benchmark_names:
            benchmark = self.get_benchmark(name)
            if benchmark:
                total_samples += benchmark.num_samples or 0
                max_memory = max(max_memory, benchmark.memory_requirements_gb or 0)
                total_time += benchmark.estimated_runtime_minutes or 30
                required_languages.update(benchmark.languages)
        
        return {
            "total_samples": total_samples,
            "max_memory_gb": max_memory,
            "estimated_runtime_minutes": total_time,
            "required_languages": list(required_languages),
            "num_benchmarks": len(benchmark_names)
        }
    
    def validate_benchmark_compatibility(self, 
                                       model_type: str, 
                                       benchmark_names: List[str]) -> Dict[str, List[str]]:
        """验证基准测试兼容性"""
        compatible = []
        incompatible = []
        warnings = []
        
        for name in benchmark_names:
            benchmark = self.get_benchmark(name)
            if not benchmark:
                incompatible.append(f"Benchmark not found: {name}")
                continue
                
            # 检查模型类型兼容性
            if model_type == "text_generation":
                if benchmark.benchmark_type in [BenchmarkType.TEXT_GENERATION, 
                                              BenchmarkType.CODE_GENERATION,
                                              BenchmarkType.REASONING]:
                    compatible.append(name)
                else:
                    warnings.append(f"May not be optimal for {name}")
            elif model_type == "classification":
                if benchmark.benchmark_type == BenchmarkType.LANGUAGE_UNDERSTANDING:
                    compatible.append(name)
                else:
                    incompatible.append(f"Not suitable for {name}")
            else:
                compatible.append(name)  # 默认兼容
        
        return {
            "compatible": compatible,
            "incompatible": incompatible,
            "warnings": warnings
        }
    
    def export_benchmark_config(self, benchmark_names: List[str], output_path: str):
        """导出基准测试配置"""
        config = {
            "benchmarks": [],
            "metadata": {
                "created_at": str(utc_now()),
                "total_benchmarks": len(benchmark_names)
            }
        }
        
        for name in benchmark_names:
            benchmark = self.get_benchmark(name)
            if benchmark:
                config["benchmarks"].append(asdict(benchmark))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
        logger.info(f"Benchmark config exported to: {output_path}")
    
    def get_recommended_benchmarks(self, 
                                 model_size: str,
                                 use_case: str,
                                 time_budget_minutes: int = 120) -> List[BenchmarkInfo]:
        """获取推荐的基准测试"""
        recommendations = []
        
        # 根据模型大小调整
        if model_size == "small":  # <1B params
            difficulty_filter = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM]
        elif model_size == "medium":  # 1-10B params  
            difficulty_filter = [DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
        else:  # >10B params
            difficulty_filter = list(DifficultyLevel)
        
        # 根据用例选择类型
        type_mapping = {
            "general": [BenchmarkType.LANGUAGE_UNDERSTANDING, BenchmarkType.REASONING],
            "code": [BenchmarkType.CODE_GENERATION],
            "qa": [BenchmarkType.QUESTION_ANSWERING, BenchmarkType.KNOWLEDGE],
            "generation": [BenchmarkType.TEXT_GENERATION, BenchmarkType.SUMMARIZATION]
        }
        
        target_types = type_mapping.get(use_case, list(BenchmarkType))
        
        # 筛选基准测试
        candidates = []
        for benchmark in self.benchmarks.values():
            if (benchmark.difficulty in difficulty_filter and 
                benchmark.benchmark_type in target_types):
                candidates.append(benchmark)
        
        # 按时间预算选择
        candidates.sort(key=lambda x: x.estimated_runtime_minutes or 30)
        total_time = 0
        
        for benchmark in candidates:
            runtime = benchmark.estimated_runtime_minutes or 30
            if total_time + runtime <= time_budget_minutes:
                recommendations.append(benchmark)
                total_time += runtime
            else:
                break
        
        return recommendations
    
    def _validate_benchmark(self, benchmark: BenchmarkInfo) -> bool:
        """验证基准测试定义是否有效"""
        try:
            # 检查必需字段
            if not benchmark.name or not benchmark.display_name:
                return False
            if not benchmark.description:
                return False
            if not benchmark.tasks or len(benchmark.tasks) == 0:
                return False
            
            # 检查枚举值
            if not isinstance(benchmark.benchmark_type, BenchmarkType):
                return False
            if not isinstance(benchmark.difficulty, DifficultyLevel):
                return False
            
            # 检查数值字段
            if benchmark.num_samples is not None and benchmark.num_samples <= 0:
                return False
            if benchmark.estimated_runtime_minutes is not None and benchmark.estimated_runtime_minutes <= 0:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Benchmark validation error: {e}")
            return False
    
    def get_suite_benchmarks(self, suite_name: str) -> List[BenchmarkInfo]:
        """获取套件中的所有基准测试"""
        suite = self.get_suite(suite_name)
        if not suite:
            return []
        
        benchmarks = []
        for benchmark_name in suite.benchmark_names:
            benchmark = self.get_benchmark(benchmark_name)
            if benchmark:
                benchmarks.append(benchmark)
        
        return benchmarks