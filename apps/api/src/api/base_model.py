"""
基础Pydantic模型
"""

from pydantic import BaseModel, ConfigDict

class ApiBaseModel(BaseModel):
    """API层统一基类"""

    model_config = ConfigDict(extra="forbid")
