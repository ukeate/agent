"""
基准测试管理器的轻量级测试
避免重依赖项导入，专注测试核心逻辑
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import json
import yaml

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

class TestBenchmarkManagerMinimal:
    """基准测试管理器轻量级测试"""
    
    def setup_method(self):
        """测试前设置"""
        from ai.model_evaluation.benchmark_manager import BenchmarkConfig
        self.config = BenchmarkConfig(
            config_path="test_benchmarks.yaml",
            cache_dir="test_cache",
            max_concurrent_downloads=2
        )
    
    def test_benchmark_config_creation(self):
        """测试基准测试配置创建"""
        assert self.config.config_path == "test_benchmarks.yaml"
        assert self.config.cache_dir == "test_cache"
        assert self.config.max_concurrent_downloads == 2
        assert self.config.enable_caching is True
        assert self.config.validation_enabled is True
    
    def test_benchmark_config_defaults(self):
        """测试基准测试配置默认值"""
        from ai.model_evaluation.benchmark_manager import BenchmarkConfig
        
        default_config = BenchmarkConfig()
        assert default_config.config_path == "config/benchmarks.yaml"
        assert default_config.cache_dir == "cache/benchmarks"
        assert default_config.max_concurrent_downloads == 3
        assert default_config.download_timeout_seconds == 300
        assert default_config.max_benchmark_size_gb == 10.0
    
    @patch('ai.model_evaluation.benchmark_manager.BenchmarkManager._initialize_standard_benchmarks')
    @patch('ai.model_evaluation.benchmark_manager.BenchmarkManager._load_custom_benchmarks')
    def test_benchmark_manager_initialization(self, mock_load_custom, mock_init_standard):
        """测试基准测试管理器初始化"""
        from ai.model_evaluation.benchmark_manager import BenchmarkManager
        
        manager = BenchmarkManager(self.config)
        
        assert manager.config == self.config
        assert manager.config_path == self.config.config_path
        assert isinstance(manager.benchmarks, dict)
        assert isinstance(manager.suites, dict)
        
        mock_init_standard.assert_called_once()
        mock_load_custom.assert_called_once()
    
    @patch('ai.model_evaluation.benchmark_manager.BenchmarkManager._initialize_standard_benchmarks')
    @patch('ai.model_evaluation.benchmark_manager.BenchmarkManager._load_custom_benchmarks')
    def test_get_benchmark_info(self, mock_load_custom, mock_init_standard):
        """测试获取基准测试信息"""
        from ai.model_evaluation.benchmark_manager import (
            BenchmarkManager, BenchmarkInfo, BenchmarkType, DifficultyLevel
        )
        
        manager = BenchmarkManager(self.config)
        
        # 添加测试基准
        test_benchmark = BenchmarkInfo(
            name="test_glue",
            display_name="Test GLUE",
            description="Test benchmark",
            benchmark_type=BenchmarkType.LANGUAGE_UNDERSTANDING,
            difficulty=DifficultyLevel.MEDIUM,
            tasks=["cola", "sst2"],
            metrics=["accuracy", "f1"]
        )
        manager.benchmarks["test_glue"] = test_benchmark
        
        # 测试获取存在的基准
        info = manager.get_benchmark("test_glue")
        assert info == test_benchmark
        assert info.name == "test_glue"
        assert info.display_name == "Test GLUE"
        
        # 测试获取不存在的基准
        info = manager.get_benchmark("nonexistent")
        assert info is None
    
    @patch('ai.model_evaluation.benchmark_manager.BenchmarkManager._initialize_standard_benchmarks')
    @patch('ai.model_evaluation.benchmark_manager.BenchmarkManager._load_custom_benchmarks')
    def test_list_benchmarks_by_type(self, mock_load_custom, mock_init_standard):
        """测试按类型列出基准测试"""
        from ai.model_evaluation.benchmark_manager import (
            BenchmarkManager, BenchmarkInfo, BenchmarkType, DifficultyLevel
        )
        
        manager = BenchmarkManager(self.config)
        
        # 添加不同类型的基准测试
        benchmarks = [
            BenchmarkInfo(
                name="glue",
                display_name="GLUE",
                description="Language Understanding",
                benchmark_type=BenchmarkType.LANGUAGE_UNDERSTANDING,
                difficulty=DifficultyLevel.MEDIUM,
                tasks=["cola", "sst2"]
            ),
            BenchmarkInfo(
                name="humaneval",
                display_name="HumanEval",
                description="Code Generation",
                benchmark_type=BenchmarkType.CODE_GENERATION,
                difficulty=DifficultyLevel.HARD,
                tasks=["humaneval"]
            ),
            BenchmarkInfo(
                name="mmlu",
                display_name="MMLU",
                description="Knowledge QA",
                benchmark_type=BenchmarkType.QUESTION_ANSWERING,
                difficulty=DifficultyLevel.EXPERT,
                tasks=["mmlu"]
            )
        ]
        
        for benchmark in benchmarks:
            manager.benchmarks[benchmark.name] = benchmark
        
        # 测试按类型筛选
        lu_benchmarks = manager.list_benchmarks(
            benchmark_type=BenchmarkType.LANGUAGE_UNDERSTANDING
        )
        assert len(lu_benchmarks) == 1
        assert lu_benchmarks[0].name == "glue"
        
        code_benchmarks = manager.list_benchmarks(
            benchmark_type=BenchmarkType.CODE_GENERATION
        )
        assert len(code_benchmarks) == 1
        assert code_benchmarks[0].name == "humaneval"
    
    @patch('ai.model_evaluation.benchmark_manager.BenchmarkManager._initialize_standard_benchmarks')
    @patch('ai.model_evaluation.benchmark_manager.BenchmarkManager._load_custom_benchmarks')
    def test_list_benchmarks_by_difficulty(self, mock_load_custom, mock_init_standard):
        """测试按难度列出基准测试"""
        from ai.model_evaluation.benchmark_manager import (
            BenchmarkManager, BenchmarkInfo, BenchmarkType, DifficultyLevel
        )
        
        manager = BenchmarkManager(self.config)
        
        # 添加不同难度的基准测试
        benchmarks = [
            BenchmarkInfo(
                name="easy_task",
                display_name="Easy Task",
                description="Simple task",
                benchmark_type=BenchmarkType.LANGUAGE_UNDERSTANDING,
                difficulty=DifficultyLevel.EASY,
                tasks=["task1"]
            ),
            BenchmarkInfo(
                name="hard_task",
                display_name="Hard Task", 
                description="Complex task",
                benchmark_type=BenchmarkType.REASONING,
                difficulty=DifficultyLevel.HARD,
                tasks=["task2"]
            )
        ]
        
        for benchmark in benchmarks:
            manager.benchmarks[benchmark.name] = benchmark
        
        # 测试按难度筛选
        easy_benchmarks = manager.list_benchmarks(difficulty=DifficultyLevel.EASY)
        assert len(easy_benchmarks) == 1
        assert easy_benchmarks[0].name == "easy_task"
        
        hard_benchmarks = manager.list_benchmarks(difficulty=DifficultyLevel.HARD)
        assert len(hard_benchmarks) == 1
        assert hard_benchmarks[0].name == "hard_task"
    
    @patch('ai.model_evaluation.benchmark_manager.BenchmarkManager._initialize_standard_benchmarks')
    @patch('ai.model_evaluation.benchmark_manager.BenchmarkManager._load_custom_benchmarks')
    def test_validate_benchmark_definition(self, mock_load_custom, mock_init_standard):
        """测试基准测试定义验证"""
        from ai.model_evaluation.benchmark_manager import (
            BenchmarkManager, BenchmarkInfo, BenchmarkType, DifficultyLevel
        )
        
        manager = BenchmarkManager(self.config)
        
        # 有效的基准测试定义
        valid_benchmark = BenchmarkInfo(
            name="valid_benchmark",
            display_name="Valid Benchmark",
            description="A valid benchmark for testing",
            benchmark_type=BenchmarkType.LANGUAGE_UNDERSTANDING,
            difficulty=DifficultyLevel.MEDIUM,
            tasks=["task1", "task2"],
            metrics=["accuracy", "f1"]
        )
        
        is_valid = manager._validate_benchmark(valid_benchmark)
        assert is_valid is True
    
    @patch('builtins.open', new_callable=mock_open, read_data='{}')
    @patch('ai.model_evaluation.benchmark_manager.BenchmarkManager._initialize_standard_benchmarks')
    def test_load_custom_benchmarks_empty_file(self, mock_init_standard, mock_file):
        """测试加载空的自定义基准测试文件"""
        from ai.model_evaluation.benchmark_manager import BenchmarkManager
        
        manager = BenchmarkManager(self.config)
        
        # 应该正常处理空文件，不抛出异常
        assert isinstance(manager.benchmarks, dict)
        assert isinstance(manager.suites, dict)
    
    @patch('builtins.open', new_callable=mock_open, read_data='invalid_yaml_content[{]')
    @patch('ai.model_evaluation.benchmark_manager.BenchmarkManager._initialize_standard_benchmarks')
    def test_load_custom_benchmarks_invalid_yaml(self, mock_init_standard, mock_file):
        """测试加载无效YAML格式的自定义基准测试文件"""
        from ai.model_evaluation.benchmark_manager import BenchmarkManager
        
        # 应该优雅地处理无效YAML，不崩溃
        manager = BenchmarkManager(self.config)
        assert isinstance(manager.benchmarks, dict)
        assert isinstance(manager.suites, dict)
    
    @patch('ai.model_evaluation.benchmark_manager.BenchmarkManager._initialize_standard_benchmarks')
    @patch('ai.model_evaluation.benchmark_manager.BenchmarkManager._load_custom_benchmarks')
    def test_create_benchmark_suite(self, mock_load_custom, mock_init_standard):
        """测试创建基准测试套件"""
        from ai.model_evaluation.benchmark_manager import (
            BenchmarkManager, BenchmarkSuite, BenchmarkInfo, BenchmarkType, DifficultyLevel
        )
        
        manager = BenchmarkManager(self.config)
        
        # 添加基准测试
        benchmarks = [
            BenchmarkInfo(
                name="glue",
                display_name="GLUE",
                description="Language Understanding",
                benchmark_type=BenchmarkType.LANGUAGE_UNDERSTANDING,
                difficulty=DifficultyLevel.MEDIUM,
                tasks=["cola", "sst2"]
            ),
            BenchmarkInfo(
                name="superglue", 
                display_name="SuperGLUE",
                description="Advanced Language Understanding",
                benchmark_type=BenchmarkType.LANGUAGE_UNDERSTANDING,
                difficulty=DifficultyLevel.HARD,
                tasks=["boolq", "cb"]
            )
        ]
        
        for benchmark in benchmarks:
            manager.benchmarks[benchmark.name] = benchmark
        
        # 创建测试套件
        suite = BenchmarkSuite(
            name="language_suite",
            display_name="Language Understanding Suite",
            description="Comprehensive language understanding evaluation",
            benchmark_names=["glue", "superglue"]
        )
        
        manager.suites["language_suite"] = suite
        
        # 验证套件创建
        retrieved_suite = manager.get_suite("language_suite")
        assert retrieved_suite == suite
        assert retrieved_suite.name == "language_suite"
        assert len(retrieved_suite.benchmark_names) == 2
    
    @patch('ai.model_evaluation.benchmark_manager.BenchmarkManager._initialize_standard_benchmarks')
    @patch('ai.model_evaluation.benchmark_manager.BenchmarkManager._load_custom_benchmarks')
    def test_get_suite_benchmarks(self, mock_load_custom, mock_init_standard):
        """测试获取套件中的基准测试"""
        from ai.model_evaluation.benchmark_manager import (
            BenchmarkManager, BenchmarkSuite, BenchmarkInfo, BenchmarkType, DifficultyLevel
        )
        
        manager = BenchmarkManager(self.config)
        
        # 添加基准测试
        benchmark1 = BenchmarkInfo(
            name="benchmark1",
            display_name="Benchmark 1",
            description="Test benchmark 1",
            benchmark_type=BenchmarkType.LANGUAGE_UNDERSTANDING,
            difficulty=DifficultyLevel.EASY,
            tasks=["task1"]
        )
        benchmark2 = BenchmarkInfo(
            name="benchmark2",
            display_name="Benchmark 2", 
            description="Test benchmark 2",
            benchmark_type=BenchmarkType.REASONING,
            difficulty=DifficultyLevel.MEDIUM,
            tasks=["task2"]
        )
        
        manager.benchmarks["benchmark1"] = benchmark1
        manager.benchmarks["benchmark2"] = benchmark2
        
        # 创建套件
        suite = BenchmarkSuite(
            name="test_suite",
            display_name="Test Suite",
            description="Test suite",
            benchmark_names=["benchmark1", "benchmark2"]
        )
        manager.suites["test_suite"] = suite
        
        # 获取套件中的基准测试
        suite_benchmarks = manager.get_suite_benchmarks("test_suite")
        
        assert len(suite_benchmarks) == 2
        assert suite_benchmarks[0].name == "benchmark1"
        assert suite_benchmarks[1].name == "benchmark2"
    
    def test_benchmark_type_enum_values(self):
        """测试基准测试类型枚举值"""
        from ai.model_evaluation.benchmark_manager import BenchmarkType
        
        expected_types = [
            "language_understanding",
            "text_generation", 
            "question_answering",
            "summarization",
            "translation",
            "code_generation",
            "reasoning",
            "knowledge",
            "safety",
            "multimodal"
        ]
        
        actual_types = [bt.value for bt in BenchmarkType]
        assert set(actual_types) == set(expected_types)
    
    def test_difficulty_level_enum_values(self):
        """测试难度级别枚举值"""
        from ai.model_evaluation.benchmark_manager import DifficultyLevel
        
        expected_levels = ["easy", "medium", "hard", "expert"]
        actual_levels = [dl.value for dl in DifficultyLevel]
        assert set(actual_levels) == set(expected_levels)

class TestBenchmarkInfo:
    """基准测试信息测试"""
    
    def test_benchmark_info_creation(self):
        """测试基准测试信息创建"""
        from ai.model_evaluation.benchmark_manager import (
            BenchmarkInfo, BenchmarkType, DifficultyLevel
        )
        
        benchmark = BenchmarkInfo(
            name="test_benchmark",
            display_name="Test Benchmark",
            description="A test benchmark",
            benchmark_type=BenchmarkType.LANGUAGE_UNDERSTANDING,
            difficulty=DifficultyLevel.MEDIUM,
            tasks=["task1", "task2"],
            languages=["en", "zh"],
            num_samples=1000,
            metrics=["accuracy", "f1", "precision"]
        )
        
        assert benchmark.name == "test_benchmark"
        assert benchmark.display_name == "Test Benchmark"
        assert benchmark.benchmark_type == BenchmarkType.LANGUAGE_UNDERSTANDING
        assert benchmark.difficulty == DifficultyLevel.MEDIUM
        assert benchmark.tasks == ["task1", "task2"]
        assert benchmark.languages == ["en", "zh"]
        assert benchmark.num_samples == 1000
        assert benchmark.metrics == ["accuracy", "f1", "precision"]
    
    def test_benchmark_info_defaults(self):
        """测试基准测试信息默认值"""
        from ai.model_evaluation.benchmark_manager import (
            BenchmarkInfo, BenchmarkType, DifficultyLevel
        )
        
        benchmark = BenchmarkInfo(
            name="minimal_benchmark",
            display_name="Minimal Benchmark",
            description="Minimal test",
            benchmark_type=BenchmarkType.LANGUAGE_UNDERSTANDING,
            difficulty=DifficultyLevel.EASY,
            tasks=["task1"]
        )
        
        assert benchmark.languages == ["en"]  # default
        assert benchmark.num_samples is None  # default
        assert benchmark.num_fewshot == 0  # default
        assert benchmark.metrics == []  # default
        assert benchmark.requires_special_setup is False  # default

if __name__ == "__main__":
    pytest.main([__file__])