"""
核心异常类定义
"""

class BaseCustomException(Exception):
    """基础自定义异常"""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class ServiceException(BaseCustomException):
    """服务层异常"""
    pass


class RepositoryException(BaseCustomException):
    """存储库异常"""
    pass


class ValidationException(BaseCustomException):
    """验证异常"""
    pass


class EncryptionException(BaseCustomException):
    """加密异常"""
    pass


class DataIntegrityException(BaseCustomException):
    """数据完整性异常"""
    pass


class AuthenticationException(BaseCustomException):
    """认证异常"""
    pass


class AuthorizationException(BaseCustomException):
    """授权异常"""
    pass