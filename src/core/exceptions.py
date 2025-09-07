"""
统一异常处理框架 - 期权套利扫描系统

提供标准化的异常处理机制，确保金融系统的稳定性和可维护性。
"""

from typing import Optional, Dict, Any, Type
from dataclasses import dataclass
import logging
from functools import wraps


@dataclass
class ErrorContext:
    """错误上下文信息"""
    component: str
    operation: str
    timestamp: float
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


class TradingSystemError(Exception):
    """交易系统基础异常"""
    
    def __init__(self, 
                 message: str, 
                 error_code: str = "GENERIC_ERROR",
                 context: Optional[ErrorContext] = None):
        self.message = message
        self.error_code = error_code
        self.context = context
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.context:
            return f"[{self.error_code}] {self.message} (Component: {self.context.component}, Operation: {self.context.operation})"
        return f"[{self.error_code}] {self.message}"


# ==================== 数据层异常 ====================

class DataError(TradingSystemError):
    """数据层基础异常"""
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(message, "DATA_ERROR", context)


class DataSourceError(DataError):
    """数据源异常"""
    def __init__(self, message: str, source: str, context: Optional[ErrorContext] = None):
        super().__init__(f"{message} (Source: {source})", context)


class TushareAPIError(DataSourceError):
    """Tushare API异常"""
    def __init__(self, message: str, status_code: Optional[int] = None, context: Optional[ErrorContext] = None):
        if status_code:
            message = f"{message} (Status: {status_code})"
        super().__init__(message, "TUSHARE", context)


class CacheError(DataError):
    """缓存异常"""
    def __init__(self, message: str, cache_type: str, context: Optional[ErrorContext] = None):
        super().__init__(f"{message} (Cache: {cache_type})", context)


# ==================== 业务层异常 ====================

class BusinessError(TradingSystemError):
    """业务层基础异常"""
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(message, "BUSINESS_ERROR", context)


class ArbitrageError(BusinessError):
    """套利计算异常"""
    def __init__(self, message: str, strategy: str, context: Optional[ErrorContext] = None):
        super().__init__(f"{message} (Strategy: {strategy})", context)


class PricingError(ArbitrageError):
    """定价计算异常"""
    def __init__(self, message: str, model: str, context: Optional[ErrorContext] = None):
        super().__init__(f"{message} (Model: {model})", "PRICING_ERROR", context)


class RiskError(BusinessError):
    """风险管理异常"""
    def __init__(self, message: str, risk_type: str, context: Optional[ErrorContext] = None):
        super().__init__(f"{message} (RiskType: {risk_type})", context)


# ==================== 系统层异常 ====================

class SystemError(TradingSystemError):
    """系统层基础异常"""
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(message, "SYSTEM_ERROR", context)


class ConfigurationError(SystemError):
    """配置异常"""
    def __init__(self, message: str, config_key: str, context: Optional[ErrorContext] = None):
        super().__init__(f"{message} (Config: {config_key})", context)


class PerformanceError(SystemError):
    """性能异常"""
    def __init__(self, message: str, metric: str, threshold: float, context: Optional[ErrorContext] = None):
        super().__init__(f"{message} (Metric: {metric}, Threshold: {threshold})", context)


# ==================== 异常处理工具 ====================

def error_handler(logger: logging.Logger):
    """异常处理装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except TradingSystemError as e:
                # 已知的系统异常，记录并重新抛出
                logger.error(f"Trading system error: {e}")
                raise
            except Exception as e:
                # 未知异常，包装为系统异常
                error_msg = f"Unexpected error in {func.__name__}: {e}"
                logger.exception(error_msg)
                raise SystemError(error_msg) from e
        return wrapper
    return decorator


def async_error_handler(logger: logging.Logger):
    """异步异常处理装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except TradingSystemError as e:
                logger.error(f"Trading system error: {e}")
                raise
            except Exception as e:
                error_msg = f"Unexpected error in {func.__name__}: {e}"
                logger.exception(error_msg)
                raise SystemError(error_msg) from e
        return wrapper
    return decorator


def create_error_context(component: str, operation: str, **kwargs) -> ErrorContext:
    """创建错误上下文"""
    import time
    return ErrorContext(
        component=component,
        operation=operation,
        timestamp=time.time(),
        additional_info=kwargs if kwargs else None
    )


def handle_data_source_error(source: str, operation: str, error: Exception) -> DataSourceError:
    """处理数据源错误"""
    context = create_error_context(
        component=f"data_source_{source.lower()}",
        operation=operation,
        source_type=source
    )
    
    if isinstance(error, TradingSystemError):
        return error
    
    return DataSourceError(
        message=f"Failed to {operation}: {error}",
        source=source,
        context=context
    )