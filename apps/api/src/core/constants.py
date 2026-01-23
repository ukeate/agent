"""
系统常量定义
"""

# 超时相关常量（秒）
class TimeoutConstants:
    """超时时间常量"""
    
    # WebSocket和对话超时时间 - 30分钟
    WEBSOCKET_TIMEOUT_SECONDS = 1800
    CONVERSATION_TIMEOUT_SECONDS = 1800
    AGENT_RESPONSE_TIMEOUT_SECONDS = 1800
    OPENAI_CLIENT_TIMEOUT_SECONDS = 1800

# 前端超时常量（毫秒）
class FrontendTimeoutConstants:
    """前端超时时间常量（毫秒）"""
    
    # API客户端超时 - 30分钟
    API_CLIENT_TIMEOUT_MS = 1800000
    
    # WebSocket相关超时
    WEBSOCKET_TIMEOUT_MS = 1800000

# 对话配置常量
class ConversationConstants:
    """对话配置常量"""
    
    # 默认最大轮数
    DEFAULT_MAX_ROUNDS = 10
    
    # 默认超时时间
    DEFAULT_TIMEOUT_SECONDS = TimeoutConstants.CONVERSATION_TIMEOUT_SECONDS
    
    # 默认自动回复
    DEFAULT_AUTO_REPLY = True
    
    # 发言者选择方法
    DEFAULT_SPEAKER_SELECTION_METHOD = "auto"
    
    # 是否允许重复发言
    DEFAULT_ALLOW_REPEAT_SPEAKER = False
