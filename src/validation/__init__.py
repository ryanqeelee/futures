"""
数据验证模块

包含数据质量验证、真实性验证等功能
"""

from .data_authenticity_validator import (
    DataAuthenticityValidator,
    ProductionDataGuard,
    ValidationResult,
    production_guard,
    ensure_authentic_data
)

__all__ = [
    'DataAuthenticityValidator',
    'ProductionDataGuard', 
    'ValidationResult',
    'production_guard',
    'ensure_authentic_data'
]