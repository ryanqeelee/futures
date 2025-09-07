# 异常处理框架迁移指南

## 🎯 迁移目标
将现有的通用异常处理模式迁移到统一的异常处理框架，提高系统的稳定性和可维护性。

## 📋 迁移步骤

### 1. 导入新的异常框架
```python
from src.core.exceptions import (
    TradingSystemError, DataSourceError, ArbitrageError,
    PricingError, RiskError, SystemError, ConfigurationError,
    error_handler, async_error_handler, create_error_context,
    handle_data_source_error
)
```

### 2. 替换通用异常处理模式

**之前:**
```python
try:
    # some operation
except Exception as e:
    logger.error(f"Error: {e}")
    raise
```

**之后:**
```python
try:
    # some operation
except Exception as e:
    # 使用适当的异常类型
    raise DataSourceError(f"Failed operation: {e}", "tushare") from e
```

### 3. 使用异常处理装饰器

**同步函数:**
```python
@error_handler(logger)
def some_function():
    # 函数逻辑
```

**异步函数:**
```python
@async_error_handler(logger) 
async def some_async_function():
    # 异步函数逻辑
```

### 4. 异常类型选择指南

| 场景 | 异常类型 | 示例 |
|------|----------|------|
| 数据源错误 | `DataSourceError` | Tushare API调用失败 |
| 套利计算错误 | `ArbitrageError` | 套利策略执行失败 |
| 定价模型错误 | `PricingError` | Black-Scholes计算失败 |
| 风险管理错误 | `RiskError` | 风险计算异常 |
| 配置错误 | `ConfigurationError` | 配置参数错误 |
| 系统错误 | `SystemError` | 未知系统异常 |

### 5. 提供错误上下文
```python
try:
    # operation
    context = create_error_context(
        component="arbitrage_engine", 
        operation="scan_opportunities",
        strategy_name=strategy_name,
        symbol=symbol
    )
    
    # 使用上下文创建异常
    raise ArbitrageError("Strategy failed", strategy_name, context)
```

## 🚀 迁移优先级

### 高优先级 (立即迁移)
1. `src/engine/arbitrage_engine.py` - 核心套利引擎
2. `src/engine/risk_manager.py` - 风险管理器
3. `src/adapters/tushare_adapter.py` - 数据源适配器

### 中优先级 (本周内完成)
4. `src/strategies/` 目录下的所有策略文件
5. `src/cache/` 目录下的缓存相关文件
6. `src/ui/` 目录下的UI组件

### 低优先级 (下周完成)
7. 工具类和辅助函数
8. 测试文件中的异常处理

## 📊 迁移状态跟踪

| 文件 | 状态 | 完成度 | 备注 |
|------|------|--------|------|
| `src/engine/arbitrage_engine.py` | 🔄 进行中 | 50% | 核心异常处理迁移 |
| `src/engine/risk_manager.py` | ⏳ 待开始 | 0% |  |
| `src/adapters/tushare_adapter.py` | ⏳ 待开始 | 0% |  |

## 🧪 测试要求

迁移后需要验证：
1. ✅ 异常类型正确性
2. ✅ 错误消息清晰度
3. ✅ 异常链完整性
4. ✅ 日志记录准确性
5. ✅ 性能影响评估

## ⚠️ 注意事项

1. **不要破坏现有功能** - 确保异常迁移不影响正常业务流程
2. **保持异常链** - 使用 `from e` 保持原始异常信息
3. **适当的日志级别** - 根据异常严重性选择合适的日志级别
4. **用户友好的错误消息** - 避免暴露敏感信息
5. **性能考虑** - 异常处理不应显著影响性能

## 🔧 工具支持

已提供以下工具函数：
- `error_handler()` - 同步函数异常处理装饰器
- `async_error_handler()` - 异步函数异常处理装饰器  
- `create_error_context()` - 创建错误上下文
- `handle_data_source_error()` - 处理数据源错误

开始迁移吧！从高优先级文件开始。