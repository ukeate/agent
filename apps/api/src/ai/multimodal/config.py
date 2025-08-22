"""
多模态处理配置管理
"""

from typing import Dict, List, Any


class ModelConfig:
    """模型配置管理"""
    
    MODEL_CONFIGS = {
        "gpt-4o": {
            "capabilities": ["text", "image", "pdf"],
            "cost_per_1k_tokens": {"input": 0.005, "output": 0.015},
            "max_tokens": 4096,
            "max_image_size": 20 * 1024 * 1024,  # 20MB
            "best_for": ["high_quality", "complex_reasoning", "pdf_processing"],
            "supports_vision": True,
            "supports_file_upload": True
        },
        "gpt-4o-mini": {
            "capabilities": ["text", "image", "pdf"],
            "cost_per_1k_tokens": {"input": 0.00015, "output": 0.0006},
            "max_tokens": 16384,
            "max_image_size": 20 * 1024 * 1024,  # 20MB
            "best_for": ["cost_effective", "simple_tasks", "high_volume"],
            "supports_vision": True,
            "supports_file_upload": True
        },
        "gpt-4o-2024-11-20": {
            "capabilities": ["text", "image", "pdf"],
            "cost_per_1k_tokens": {"input": 0.00275, "output": 0.011},
            "max_tokens": 16384,
            "max_image_size": 20 * 1024 * 1024,  # 20MB
            "best_for": ["latest_features", "structured_output"],
            "supports_vision": True,
            "supports_file_upload": True
        },
        "gpt-5": {
            "capabilities": ["text", "image", "pdf", "video"],
            "cost_per_1k_tokens": {"input": 0.0125, "output": 0.025},
            "max_tokens": 8192,
            "max_image_size": 50 * 1024 * 1024,  # 50MB
            "best_for": ["latest_features", "advanced_reasoning", "complex_multimodal"],
            "supports_vision": True,
            "supports_file_upload": True
        },
        "gpt-5-mini": {
            "capabilities": ["text", "image", "pdf"],
            "cost_per_1k_tokens": {"input": 0.0025, "output": 0.005},
            "max_tokens": 16384,
            "max_image_size": 30 * 1024 * 1024,  # 30MB
            "best_for": ["balanced_performance", "moderate_complexity"],
            "supports_vision": True,
            "supports_file_upload": True
        },
        "gpt-5-nano": {
            "capabilities": ["text", "image"],
            "cost_per_1k_tokens": {"input": 0.00005, "output": 0.0004},
            "max_tokens": 128000,
            "max_image_size": 10 * 1024 * 1024,  # 10MB
            "best_for": ["summarization", "classification", "high_volume", "ultra_low_cost"],
            "supports_vision": True,
            "supports_file_upload": False
        }
    }
    
    # 支持的文件格式
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
    SUPPORTED_DOCUMENT_FORMATS = {'.pdf', '.txt', '.docx', '.md', '.csv', '.xlsx'}
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    SUPPORTED_AUDIO_FORMATS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
    
    # API配置
    OPENAI_BASE_URL = "https://api.openai.com/v1"
    FILE_UPLOAD_ENDPOINT = "/files"
    CHAT_COMPLETION_ENDPOINT = "/chat/completions"
    
    # 处理限制
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_CONCURRENT_REQUESTS = 10
    DEFAULT_TIMEOUT = 300  # 5分钟
    
    # 缓存配置
    CACHE_TTL = 3600  # 1小时
    CACHE_KEY_PREFIX = "multimodal:"
    
    # 视频处理配置
    MAX_VIDEO_FRAMES = 10
    FRAME_EXTRACTION_INTERVAL = 2.0  # 秒
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """获取模型配置"""
        return cls.MODEL_CONFIGS.get(model_name, cls.MODEL_CONFIGS["gpt-4o-mini"])
    
    @classmethod
    def get_supported_formats(cls, content_type: str) -> set:
        """获取支持的文件格式"""
        format_map = {
            "image": cls.SUPPORTED_IMAGE_FORMATS,
            "document": cls.SUPPORTED_DOCUMENT_FORMATS,
            "video": cls.SUPPORTED_VIDEO_FORMATS,
            "audio": cls.SUPPORTED_AUDIO_FORMATS
        }
        return format_map.get(content_type, set())
    
    @classmethod
    def is_format_supported(cls, file_extension: str, content_type: str) -> bool:
        """检查文件格式是否支持"""
        supported_formats = cls.get_supported_formats(content_type)
        return file_extension.lower() in supported_formats