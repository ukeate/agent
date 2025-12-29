import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open
from ai.model_evaluation.benchmark_manager import (
    BenchmarkManager,
    BenchmarkInfo,
    BenchmarkSuite,
    BenchmarkType,
    DifficultyLevel
)

class TestBenchmarkInfo:
    """测试基准测试信息类"""
    
    def test_benchmark_info_creation(self):
        """测试基准测试信息创建"""
        benchmark = BenchmarkInfo(
            name="test_benchmark",
            display_name="Test Benchmark",
            description="A test benchmark",
            benchmark_type=BenchmarkType.LANGUAGE_UNDERSTANDING,
            difficulty=DifficultyLevel.MEDIUM,
            tasks=["task1", "task2"],
            languages=["en", "zh"],
            num_samples=1000,
            metrics=["accuracy", "f1"]
        )
        
        assert benchmark.name == "test_benchmark"
        assert benchmark.display_name == "Test Benchmark"
        assert benchmark.description == "A test benchmark"
        assert benchmark.benchmark_type == BenchmarkType.LANGUAGE_UNDERSTANDING
        assert benchmark.difficulty == DifficultyLevel.MEDIUM
        assert benchmark.tasks == ["task1", "task2"]
        assert benchmark.languages == ["en", "zh"]
        assert benchmark.num_samples == 1000
        assert benchmark.metrics == ["accuracy", "f1"]
    
    def test_benchmark_info_defaults(self):
        """测试基准测试信息默认值"""
        benchmark = BenchmarkInfo(
            name="minimal_benchmark",
            display_name="Minimal Benchmark",
            description="Minimal test",
            benchmark_type=BenchmarkType.TEXT_GENERATION,
            difficulty=DifficultyLevel.EASY,
            tasks=["task1"]
        )
        
        assert benchmark.languages == ["en"]  # 默认语言
        assert benchmark.num_samples is None
        assert benchmark.num_fewshot == 0
        assert benchmark.metrics == []
        assert not benchmark.requires_special_setup
        assert benchmark.estimated_runtime_minutes is None
        assert benchmark.memory_requirements_gb is None

class TestBenchmarkSuite:
    """测试基准测试套件类"""
    
    def test_benchmark_suite_creation(self):
        """测试基准测试套件创建"""
        benchmarks = [
            BenchmarkInfo(
                name="bench1",
                display_name="Benchmark 1",
                description="First benchmark",
                benchmark_type=BenchmarkType.REASONING,
                difficulty=DifficultyLevel.HARD,
                tasks=["task1"],
                estimated_runtime_minutes=30
            ),
            BenchmarkInfo(
                name="bench2",
                display_name="Benchmark 2", 
                description="Second benchmark",
                benchmark_type=BenchmarkType.CODE_GENERATION,
                difficulty=DifficultyLevel.MEDIUM,
                tasks=["task2"],
                estimated_runtime_minutes=45
            )
        ]
        
        suite = BenchmarkSuite(
            name="test_suite",
            description="Test suite",
            benchmarks=benchmarks
        )
        
        assert suite.name == "test_suite"
        assert suite.description == "Test suite"
        assert suite.total_tasks == 2
        assert suite.estimated_runtime_hours == (30 + 45) / 60.0
        assert len(suite.benchmarks) == 2
    
    def test_benchmark_suite_post_init(self):
        """测试基准测试套件后初始化"""
        benchmarks = [
            BenchmarkInfo(
                name="bench1",
                display_name="Benchmark 1",
                description="First benchmark",
                benchmark_type=BenchmarkType.REASONING,
                difficulty=DifficultyLevel.HARD,
                tasks=["task1"]
            )
        ]
        
        suite = BenchmarkSuite(
            name="auto_calc_suite",
            description="Auto calculation test",
            benchmarks=benchmarks
        )
        
        # __post_init__ 应该自动计算这些值
        assert suite.total_tasks == 1
        assert suite.estimated_runtime_hours == 30 / 60.0  # 默认30分钟

class TestBenchmarkManager:
    """测试基准测试管理器"""
    
    @pytest.fixture
    def benchmark_manager(self):
        """创建基准测试管理器实例"""
        with patch.object(BenchmarkManager, '_load_custom_benchmarks'):
            manager = BenchmarkManager()
            return manager
    
    @pytest.fixture
    def temp_config_file(self):
        """创建临时配置文件"""
        config_data = {
            "benchmarks": [
                {
                    "name": "custom_benchmark",
                    "display_name": "Custom Benchmark",
                    "description": "A custom test benchmark",
                    "benchmark_type": "text_generation",
                    "difficulty": "easy",
                    "tasks": ["custom_task"],
                    "languages": ["en"],
                    "num_samples": 500,
                    "metrics": ["accuracy", "bleu"]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = Path(f.name)
        
        yield temp_file
        temp_file.unlink()  # 清理临时文件
    
    def test_manager_initialization(self, benchmark_manager):
        """测试管理器初始化"""
        assert isinstance(benchmark_manager.benchmarks, dict)
        assert isinstance(benchmark_manager.suites, dict)
        
        # 验证标准基准测试已加载
        assert "cola" in benchmark_manager.benchmarks
        assert "sst2" in benchmark_manager.benchmarks
        assert "mmlu" in benchmark_manager.benchmarks
        assert "humaneval" in benchmark_manager.benchmarks
        assert "hellaswag" in benchmark_manager.benchmarks
        
        # 验证套件已创建
        assert "glue" in benchmark_manager.suites
        assert "superglue" in benchmark_manager.suites
        assert "code" in benchmark_manager.suites
        assert "reasoning" in benchmark_manager.suites
        assert "comprehensive" in benchmark_manager.suites
    
    def test_get_benchmark(self, benchmark_manager):
        """测试获取单个基准测试"""
        # 获取存在的基准测试
        cola = benchmark_manager.get_benchmark("cola")
        assert cola is not None
        assert cola.name == "cola"
        assert cola.display_name == "CoLA"
        assert cola.benchmark_type == BenchmarkType.LANGUAGE_UNDERSTANDING
        
        # 获取不存在的基准测试
        nonexistent = benchmark_manager.get_benchmark("nonexistent")
        assert nonexistent is None
    
    def test_get_benchmarks_by_type(self, benchmark_manager):
        """测试按类型获取基准测试"""
        # 获取语言理解类基准测试
        lu_benchmarks = benchmark_manager.get_benchmarks_by_type(BenchmarkType.LANGUAGE_UNDERSTANDING)
        
        assert len(lu_benchmarks) > 0
        assert all(b.benchmark_type == BenchmarkType.LANGUAGE_UNDERSTANDING for b in lu_benchmarks)
        
        # 验证包含预期的基准测试
        benchmark_names = [b.name for b in lu_benchmarks]
        assert "cola" in benchmark_names
        assert "sst2" in benchmark_names
        
        # 获取代码生成类基准测试
        code_benchmarks = benchmark_manager.get_benchmarks_by_type(BenchmarkType.CODE_GENERATION)
        code_names = [b.name for b in code_benchmarks]
        assert "humaneval" in code_names
        assert "mbpp" in code_names
    
    def test_get_benchmarks_by_difficulty(self, benchmark_manager):
        """测试按难度获取基准测试"""
        # 获取简单难度的基准测试
        easy_benchmarks = benchmark_manager.get_benchmarks_by_difficulty(DifficultyLevel.EASY)
        
        assert len(easy_benchmarks) > 0
        assert all(b.difficulty == DifficultyLevel.EASY for b in easy_benchmarks)
        
        # 获取困难难度的基准测试
        hard_benchmarks = benchmark_manager.get_benchmarks_by_difficulty(DifficultyLevel.HARD)
        hard_names = [b.name for b in hard_benchmarks]
        assert "mmlu" in hard_names
        assert "humaneval" in hard_names
    
    def test_get_suite(self, benchmark_manager):
        """测试获取基准测试套件"""
        # 获取存在的套件
        glue_suite = benchmark_manager.get_suite("glue")
        assert glue_suite is not None
        assert glue_suite.name == "glue"
        assert glue_suite.description == "General Language Understanding Evaluation"
        assert len(glue_suite.benchmarks) > 0
        
        # 获取不存在的套件
        nonexistent_suite = benchmark_manager.get_suite("nonexistent")
        assert nonexistent_suite is None
    
    def test_list_benchmarks_with_filters(self, benchmark_manager):
        """测试带过滤条件的基准测试列表"""
        # 测试类型过滤
        code_benchmarks = benchmark_manager.list_benchmarks(
            benchmark_type=BenchmarkType.CODE_GENERATION
        )
        assert len(code_benchmarks) > 0
        assert all(b.benchmark_type == BenchmarkType.CODE_GENERATION for b in code_benchmarks)
        
        # 测试难度过滤
        easy_benchmarks = benchmark_manager.list_benchmarks(
            difficulty=DifficultyLevel.EASY
        )
        assert len(easy_benchmarks) > 0
        assert all(b.difficulty == DifficultyLevel.EASY for b in easy_benchmarks)
        
        # 测试语言过滤
        en_benchmarks = benchmark_manager.list_benchmarks(language="en")
        assert len(en_benchmarks) > 0
        assert all("en" in b.languages for b in en_benchmarks)
        
        # 测试组合过滤
        filtered_benchmarks = benchmark_manager.list_benchmarks(
            benchmark_type=BenchmarkType.LANGUAGE_UNDERSTANDING,
            difficulty=DifficultyLevel.MEDIUM,
            language="en"
        )
        
        for benchmark in filtered_benchmarks:
            assert benchmark.benchmark_type == BenchmarkType.LANGUAGE_UNDERSTANDING
            assert benchmark.difficulty == DifficultyLevel.MEDIUM
            assert "en" in benchmark.languages
    
    def test_list_suites(self, benchmark_manager):
        """测试列出所有套件"""
        suites = benchmark_manager.list_suites()
        
        assert len(suites) >= 5  # 至少有预定义的5个套件
        suite_names = [s.name for s in suites]
        assert "glue" in suite_names
        assert "superglue" in suite_names
        assert "code" in suite_names
        assert "reasoning" in suite_names
        assert "comprehensive" in suite_names
    
    def test_create_custom_suite(self, benchmark_manager):
        """测试创建自定义套件"""
        benchmark_names = ["cola", "sst2", "hellaswag"]
        
        custom_suite = benchmark_manager.create_custom_suite(
            name="my_custom_suite",
            description="My custom benchmark suite",
            benchmark_names=benchmark_names
        )
        
        assert custom_suite.name == "my_custom_suite"
        assert custom_suite.description == "My custom benchmark suite"
        assert len(custom_suite.benchmarks) == 3
        
        # 验证套件已添加到管理器
        assert "my_custom_suite" in benchmark_manager.suites
        
        retrieved_suite = benchmark_manager.get_suite("my_custom_suite")
        assert retrieved_suite is not None
        assert retrieved_suite.name == "my_custom_suite"
    
    def test_create_custom_suite_with_invalid_benchmark(self, benchmark_manager):
        """测试使用无效基准测试名称创建自定义套件"""
        benchmark_names = ["cola", "invalid_benchmark", "sst2"]
        
        with patch('logging.Logger.warning') as mock_warning:
            custom_suite = benchmark_manager.create_custom_suite(
                name="suite_with_invalid",
                description="Suite with invalid benchmark",
                benchmark_names=benchmark_names
            )
            
            # 应该只包含有效的基准测试
            assert len(custom_suite.benchmarks) == 2
            benchmark_names_in_suite = [b.name for b in custom_suite.benchmarks]
            assert "cola" in benchmark_names_in_suite
            assert "sst2" in benchmark_names_in_suite
            assert "invalid_benchmark" not in benchmark_names_in_suite
            
            # 应该记录警告
            mock_warning.assert_called_with("Benchmark not found: invalid_benchmark")
    
    def test_get_benchmark_requirements(self, benchmark_manager):
        """测试获取基准测试资源需求"""
        benchmark_names = ["cola", "sst2", "mmlu"]
        
        requirements = benchmark_manager.get_benchmark_requirements(benchmark_names)
        
        assert "total_samples" in requirements
        assert "max_memory_gb" in requirements
        assert "estimated_runtime_minutes" in requirements
        assert "required_languages" in requirements
        assert "num_benchmarks" in requirements
        
        assert requirements["num_benchmarks"] == 3
        assert requirements["total_samples"] > 0  # 应该有样本统计
        assert "en" in requirements["required_languages"]
    
    def test_validate_benchmark_compatibility(self, benchmark_manager):
        """测试验证基准测试兼容性"""
        # 测试文本生成模型兼容性
        benchmark_names = ["humaneval", "mbpp", "cola"]
        
        compatibility = benchmark_manager.validate_benchmark_compatibility(
            model_type="text_generation",
            benchmark_names=benchmark_names
        )
        
        assert "compatible" in compatibility
        assert "incompatible" in compatibility
        assert "warnings" in compatibility
        
        # 代码生成基准测试应该兼容
        assert "humaneval" in compatibility["compatible"]
        assert "mbpp" in compatibility["compatible"]
        
        # 分类任务可能有警告
        assert "cola" in compatibility["warnings"] or "cola" in compatibility["compatible"]
        
        # 测试分类模型兼容性
        compatibility_classification = benchmark_manager.validate_benchmark_compatibility(
            model_type="classification",
            benchmark_names=["cola", "sst2", "humaneval"]
        )
        
        # 语言理解任务应该兼容
        assert "cola" in compatibility_classification["compatible"]
        assert "sst2" in compatibility_classification["compatible"]
        
        # 代码生成任务应该不兼容
        assert "humaneval" in compatibility_classification["incompatible"]
    
    def test_get_recommended_benchmarks(self, benchmark_manager):
        """测试获取推荐的基准测试"""
        # 测试小模型推荐
        small_model_recommendations = benchmark_manager.get_recommended_benchmarks(
            model_size="small",
            use_case="general",
            time_budget_minutes=60
        )
        
        assert len(small_model_recommendations) > 0
        
        # 验证难度级别
        for benchmark in small_model_recommendations:
            assert benchmark.difficulty in [DifficultyLevel.EASY, DifficultyLevel.MEDIUM]
        
        # 验证时间预算
        total_time = sum(b.estimated_runtime_minutes or 30 for b in small_model_recommendations)
        assert total_time <= 60
        
        # 测试代码用例推荐
        code_recommendations = benchmark_manager.get_recommended_benchmarks(
            model_size="large",
            use_case="code",
            time_budget_minutes=120
        )
        
        # 应该主要包含代码生成基准测试
        code_benchmarks = [b for b in code_recommendations 
                          if b.benchmark_type == BenchmarkType.CODE_GENERATION]
        assert len(code_benchmarks) > 0
    
    def test_export_benchmark_config(self, benchmark_manager, temp_config_file):
        """测试导出基准测试配置"""
        benchmark_names = ["cola", "sst2"]
        output_path = str(temp_config_file)
        
        benchmark_manager.export_benchmark_config(benchmark_names, output_path)
        
        # 验证文件已创建并包含正确内容
        assert temp_config_file.exists()
        
        with open(temp_config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        assert "benchmarks" in config_data
        assert "metadata" in config_data
        assert len(config_data["benchmarks"]) == 2
        assert config_data["metadata"]["total_benchmarks"] == 2
        
        # 验证基准测试数据
        benchmark_names_in_export = [b["name"] for b in config_data["benchmarks"]]
        assert "cola" in benchmark_names_in_export
        assert "sst2" in benchmark_names_in_export
    
    def test_load_custom_benchmarks(self, temp_config_file):
        """测试加载自定义基准测试"""
        # 创建包含自定义基准测试的配置文件
        config_data = {
            "benchmarks": [
                {
                    "name": "custom_benchmark",
                    "display_name": "Custom Benchmark",
                    "description": "A custom test benchmark",
                    "benchmark_type": "text_generation",
                    "difficulty": "easy",
                    "tasks": ["custom_task"],
                    "languages": ["en"],
                    "num_samples": 500,
                    "metrics": ["accuracy", "bleu"]
                }
            ]
        }
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # 创建管理器并加载自定义配置
        with patch('logging.Logger.info') as mock_info:
            manager = BenchmarkManager(config_path=str(temp_config_file))
            
            # 验证自定义基准测试已加载
            assert "custom_benchmark" in manager.benchmarks
            
            custom_benchmark = manager.get_benchmark("custom_benchmark")
            assert custom_benchmark is not None
            assert custom_benchmark.display_name == "Custom Benchmark"
            assert custom_benchmark.benchmark_type == BenchmarkType.TEXT_GENERATION
            assert custom_benchmark.difficulty == DifficultyLevel.EASY
            assert custom_benchmark.num_samples == 500
            
            # 验证日志记录
            mock_info.assert_called_with("Loaded custom benchmark: custom_benchmark")
    
    def test_load_custom_benchmarks_file_not_found(self):
        """测试加载不存在的自定义配置文件"""
        nonexistent_path = "nonexistent_config.yaml"
        
        # 应该不会抛出异常，只是没有加载自定义基准测试
        manager = BenchmarkManager(config_path=nonexistent_path)
        
        # 仍然应该有标准基准测试
        assert "cola" in manager.benchmarks
        assert "sst2" in manager.benchmarks
    
    def test_load_custom_benchmarks_invalid_yaml(self, temp_config_file):
        """测试加载无效的YAML配置文件"""
        # 写入无效的YAML
        with open(temp_config_file, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with patch('logging.Logger.warning') as mock_warning:
            manager = BenchmarkManager(config_path=str(temp_config_file))
            
            # 应该记录警告并继续使用标准基准测试
            mock_warning.assert_called()
            assert "cola" in manager.benchmarks

if __name__ == "__main__":
    pytest.main([__file__])
