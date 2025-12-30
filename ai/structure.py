from pydantic import BaseModel, Field, field_validator
from typing import Optional
import re


class Structure(BaseModel):
    """论文结构化信息模型，带有宽容的字段验证"""
    
    tldr: str = Field(
        default="Summary not available",
        description="generate a too long; didn't read summary"
    )
    motivation: str = Field(
        default="Motivation not available",
        description="describe the motivation in this paper"
    )
    method: str = Field(
        default="Method not available",
        description="method of this paper"
    )
    result: str = Field(
        default="Result not available",
        description="result of this paper"
    )
    conclusion: str = Field(
        default="Conclusion not available",
        description="conclusion of this paper"
    )

    @field_validator('*', mode='before')
    @classmethod
    def clean_and_validate_field(cls, v):
        """清理并验证字段值，处理各种边界情况"""
        if v is None:
            return ""
        if not isinstance(v, str):
            v = str(v)
        
        # 去除首尾空白
        v = v.strip()
        
        # 处理多余的换行符（保留单个换行作为段落分隔）
        v = re.sub(r'\n{3,}', '\n\n', v)
        
        # 移除控制字符（保留换行和制表符）
        v = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', v)
        
        return v if v else ""

    class Config:
        # 允许额外字段，避免解析失败
        extra = 'ignore'
        # 字符串自动去除空白
        str_strip_whitespace = True