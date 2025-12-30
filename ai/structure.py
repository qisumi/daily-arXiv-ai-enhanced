from pydantic import BaseModel, Field, field_validator
from typing import Optional
import re


class Structure(BaseModel):
    """论文结构化信息模型，带有宽容的字段验证"""
    
    tldr: str = Field(
        default="摘要生成失败",
        description="生成论文的简短摘要（TL;DR）/ Generate a concise summary (TL;DR) of the paper"
    )
    motivation: str = Field(
        default="动机分析不可用",
        description="描述论文的研究动机和背景 / Describe the research motivation and background"
    )
    method: str = Field(
        default="方法提取失败",
        description="提取论文的研究方法 / Extract the research method used"
    )
    result: str = Field(
        default="结果分析不可用",
        description="总结论文的主要结果 / Summarize the main results"
    )
    conclusion: str = Field(
        default="结论提取失败",
        description="提取论文的结论 / Extract the conclusion"
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