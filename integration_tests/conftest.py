"""
集成测试配置文件 - 期权套利扫描系统

提供测试夹具、模拟数据和测试环境配置
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
import yaml
import os
import sys

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# 导入系统组件
from src.config.models import (
    ArbitrageOpportunity, StrategyType, RiskLevel, SystemConfig, RiskConfig
)
from src.strategies.base import (
    OptionData, OptionType, BaseStrategy, StrategyResult
)
from src.adapters.base import DataResponse
from src.engine.arbitrage_engine import ArbitrageEngine, ScanParameters
from src.engine.risk_manager import AdvancedRiskManager


# ==================== 测试环境配置 ====================

@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环供整个测试会话使用"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(scope="session") 
def test_temp_dir():
    """创建临时测试目录"""
    temp_dir = tempfile.mkdtemp(prefix="futures_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def test_config_dir(test_temp_dir):
    """创建测试配置目录"""
    config_dir = test_temp_dir / "config"
    config_dir.mkdir(exist_ok=True)
    
    # 创建测试配置文件
    config_content = {
        'app_name': 'Test Options Arbitrage Scanner',
        'version': '1.0.0',
        'environment': 'test',
        'debug': True,
        'log_level': 'INFO',
        'data_dir': str(test_temp_dir / 'data'),
        'log_dir': str(test_temp_dir / 'logs'),
        'config_dir': str(config_dir),
        'risk': {
            'max_position_size': 10000.0,
            'max_daily_loss': 1000.0,
            'min_liquidity_volume': 100,
            'max_concentration': 0.3,
            'max_days_to_expiry': 90,
            'min_days_to_expiry': 1
        },
        'data_sources': {
            'tushare': {
                'type': 'tushare',
                'enabled': True,
                'priority': 1,
                'api_token': 'test_token_12345',
                'rate_limit': 200,
                'timeout': 30
            }
        },
        'strategies': {
            'pricing_arbitrage': {
                'type': 'pricing_arbitrage',
                'enabled': True,
                'priority': 1,
                'min_profit_threshold': 0.02,
                'max_risk_tolerance': 0.15
            },
            'put_call_parity': {
                'type': 'put_call_parity',
                'enabled': True,
                'priority': 2,
                'tolerance': 0.05
            },
            'volatility_arbitrage': {
                'type': 'volatility_arbitrage',
                'enabled': True,
                'priority': 3,
                'implied_vol_threshold': 0.1
            }
        },
        'cache': {
            'type': 'memory',
            'max_size': 1000,
            'ttl_seconds': 300,
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 1
        },
        'monitoring': {
            'enable_performance_monitoring': True,
            'metrics_collection_interval': 60,
            'alert_thresholds': {
                'max_response_time': 5.0,
                'min_success_rate': 0.95
            }
        }
    }
    
    config_file = config_dir / "config.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f, default_flow_style=False, allow_unicode=True)
    
    # 创建其他必要目录
    (test_temp_dir / 'data').mkdir(exist_ok=True)
    (test_temp_dir / 'logs').mkdir(exist_ok=True)
    
    return config_dir


# ==================== 数据夹具 ====================

@pytest.fixture
def sample_option_data():
    """生成样本期权数据"""
    np.random.seed(42)  # 确保可重现性
    
    options = []
    underlyings = ['IF2312', 'IH2312', 'IC2312', 'IM2312', 'MO2312']
    
    for underlying in underlyings:
        base_price = np.random.uniform(3800, 4200)
        for i in range(20):  # 每个标的20个期权
            strike = base_price + (i - 10) * 50  # 围绕基础价格的行权价
            
            for option_type in [OptionType.CALL, OptionType.PUT]:
                expiry_days = np.random.randint(5, 60)
                
                option = OptionData(
                    code=f"{underlying}-{option_type.value}-{int(strike):04d}",
                    name=f"{underlying} {option_type.value} {int(strike)}",
                    underlying=underlying,
                    option_type=option_type,
                    strike_price=float(strike),
                    expiry_date=datetime.now() + timedelta(days=expiry_days),
                    market_price=max(0.1, np.random.gamma(2, 25)),  # 伽马分布生成价格
                    bid_price=None,  # 会在后续填充
                    ask_price=None,  # 会在后续填充
                    volume=max(10, int(np.random.exponential(100))),
                    open_interest=max(1, int(np.random.exponential(50))),
                    implied_volatility=max(0.1, np.random.normal(0.25, 0.1)),
                    theoretical_price=None,  # 会在后续计算
                    delta=None,
                    gamma=None,
                    theta=None,
                    vega=None,
                    underlying_price=base_price + np.random.normal(0, 10)
                )
                
                # 填充买卖价差
                spread = option.market_price * 0.02  # 2% 价差
                option.bid_price = max(0.05, option.market_price - spread/2)
                option.ask_price = option.market_price + spread/2
                
                options.append(option)
    
    return options


@pytest.fixture 
def historical_price_data():
    """生成历史价格数据用于风险计算"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # 生成具有现实特征的价格序列
    returns = np.random.normal(0.0005, 0.02, len(dates))  # 年化约12.5%收益，20%波动率
    
    price_data = {
        'date': dates,
        'IF2312': 4000 * np.exp(np.cumsum(returns + np.random.normal(0, 0.001, len(dates)))),
        'IH2312': 3800 * np.exp(np.cumsum(returns + np.random.normal(0, 0.001, len(dates)))),
        'IC2312': 4200 * np.exp(np.cumsum(returns + np.random.normal(0, 0.001, len(dates)))),
        'IM2312': 3900 * np.exp(np.cumsum(returns + np.random.normal(0, 0.001, len(dates)))),
        'MO2312': 4100 * np.exp(np.cumsum(returns + np.random.normal(0, 0.001, len(dates))))
    }
    
    return pd.DataFrame(price_data)


@pytest.fixture
def mock_arbitrage_opportunities():
    """生成模拟套利机会"""
    opportunities = [
        ArbitrageOpportunity(
            id="test_pricing_arb_001",
            strategy_type=StrategyType.PRICING_ARBITRAGE,
            instruments=["IF2312-C-4000", "IF2312-P-4000"],
            underlying="IF2312",
            expected_profit=125.50,
            profit_margin=0.0314,
            confidence_score=0.85,
            max_loss=45.20,
            risk_score=0.12,
            days_to_expiry=25,
            market_prices={
                "IF2312-C-4000": 89.5,
                "IF2312-P-4000": 76.2
            },
            volumes={
                "IF2312-C-4000": 850,
                "IF2312-P-4000": 720
            },
            actions=[
                {"instrument": "IF2312-C-4000", "action": "BUY", "quantity": 2, "price": 89.5},
                {"instrument": "IF2312-P-4000", "action": "SELL", "quantity": 2, "price": 76.2}
            ],
            data_source="test_mock",
            discovery_time=datetime.now(),
            expiration_time=datetime.now() + timedelta(minutes=30),
            metadata={
                "strategy_specific": {
                    "iv_call": 0.24,
                    "iv_put": 0.26,
                    "underlying_price": 4025.8,
                    "time_to_expiry": 0.0685
                }
            }
        ),
        ArbitrageOpportunity(
            id="test_parity_arb_002",
            strategy_type=StrategyType.PUT_CALL_PARITY,
            instruments=["IH2312-C-3800", "IH2312-P-3800"],
            underlying="IH2312",
            expected_profit=67.30,
            profit_margin=0.0189,
            confidence_score=0.92,
            max_loss=28.15,
            risk_score=0.08,
            days_to_expiry=18,
            market_prices={
                "IH2312-C-3800": 102.3,
                "IH2312-P-3800": 58.7
            },
            volumes={
                "IH2312-C-3800": 1200,
                "IH2312-P-3800": 980
            },
            actions=[
                {"instrument": "IH2312-C-3800", "action": "BUY", "quantity": 1, "price": 102.3},
                {"instrument": "IH2312-P-3800", "action": "SELL", "quantity": 1, "price": 58.7}
            ],
            data_source="test_mock",
            discovery_time=datetime.now(),
            expiration_time=datetime.now() + timedelta(minutes=45),
            metadata={
                "parity_deviation": 0.0215,
                "risk_free_rate": 0.025,
                "dividend_yield": 0.0
            }
        )
    ]
    
    return opportunities


# ==================== 系统组件夹具 ====================

@pytest.fixture
async def mock_config_manager(test_config_dir):
    """创建模拟配置管理器"""
    from src.config.manager import ConfigManager
    
    config_manager = Mock()
    
    # 模拟系统配置
    system_config = Mock()
    system_config.app_name = "Test Options Arbitrage Scanner"
    system_config.environment = "test"
    system_config.debug = True
    
    # 风险配置
    system_config.risk = RiskConfig(
        max_position_size=10000.0,
        max_daily_loss=1000.0,
        min_liquidity_volume=100,
        max_concentration=0.3,
        max_days_to_expiry=90,
        min_days_to_expiry=1
    )
    
    config_manager.get_system_config.return_value = system_config
    config_manager.config_dir = test_config_dir
    
    return config_manager


@pytest.fixture
def mock_data_adapter(sample_option_data):
    """创建模拟数据适配器"""
    from src.adapters.base import BaseDataAdapter, DataRequest, DataResponse
    
    adapter = AsyncMock(spec=BaseDataAdapter)
    adapter.data_source_type = "test_mock"
    adapter.connection_status = "CONNECTED"
    
    # 模拟数据获取
    async def mock_get_option_data(request):
        return DataResponse(
            request=request,
            data=sample_option_data[:50],  # 返回前50个期权
            timestamp=datetime.now(),
            source="test_mock",
            quality="HIGH",
            metadata={
                "total_records": len(sample_option_data),
                "filtered_records": 50,
                "data_freshness": "REAL_TIME"
            }
        )
    
    adapter.get_option_data.side_effect = mock_get_option_data
    adapter.get_underlying_price.return_value = 4000.0
    
    return adapter


@pytest.fixture 
def mock_strategy():
    """创建模拟策略"""
    strategy = Mock(spec=BaseStrategy)
    strategy.strategy_type = StrategyType.PRICING_ARBITRAGE
    strategy.name = "MockPricingArbitrageStrategy"
    strategy.enabled = True
    
    def mock_scan_opportunities(options_data):
        if not options_data:
            return StrategyResult(
                strategy_name="MockPricingArbitrageStrategy",
                opportunities=[],
                execution_time=0.05,
                data_timestamp=datetime.now(),
                success=True,
                metadata={"scanned_options": 0}
            )
        
        # 生成一些模拟机会
        opportunities = []
        for i, option in enumerate(options_data[:3]):  # 只处理前3个
            opp = ArbitrageOpportunity(
                id=f"mock_opp_{i+1}",
                strategy_type=StrategyType.PRICING_ARBITRAGE,
                instruments=[option.code],
                underlying=option.underlying,
                expected_profit=float(50 + i * 25),
                profit_margin=0.02 + i * 0.01,
                confidence_score=0.8 - i * 0.05,
                max_loss=float(20 + i * 5),
                risk_score=0.1 + i * 0.05,
                days_to_expiry=option.expiry_date.day,
                market_prices={option.code: option.market_price},
                volumes={option.code: option.volume},
                actions=[],
                data_source="mock_strategy"
            )
            opportunities.append(opp)
        
        return StrategyResult(
            strategy_name="MockPricingArbitrageStrategy",
            opportunities=opportunities,
            execution_time=0.15,
            data_timestamp=datetime.now(),
            success=True,
            metadata={"scanned_options": len(options_data)}
        )
    
    strategy.scan_opportunities.side_effect = mock_scan_opportunities
    return strategy


@pytest.fixture
async def integration_arbitrage_engine(
    mock_config_manager, 
    mock_data_adapter, 
    mock_strategy
):
    """创建集成测试用的套利引擎"""
    data_adapters = {"test_mock": mock_data_adapter}
    strategies = {StrategyType.PRICING_ARBITRAGE: mock_strategy}
    
    engine = ArbitrageEngine(
        config_manager=mock_config_manager,
        data_adapters=data_adapters,
        strategies=strategies
    )
    
    yield engine
    
    # 清理
    await engine.shutdown()


@pytest.fixture
def risk_manager(mock_config_manager):
    """创建风险管理器"""
    risk_config = mock_config_manager.get_system_config.return_value.risk
    return AdvancedRiskManager(risk_config)


# ==================== 测试数据工厂 ====================

class TestDataFactory:
    """测试数据工厂类"""
    
    @staticmethod
    def create_option_chain(underlying: str, base_price: float, expiry_days: int = 30) -> List[OptionData]:
        """创建期权链"""
        options = []
        strikes = [base_price + (i - 5) * 50 for i in range(11)]  # 11个行权价
        
        for strike in strikes:
            for option_type in [OptionType.CALL, OptionType.PUT]:
                option = OptionData(
                    code=f"{underlying}-{option_type.value}-{int(strike)}",
                    name=f"{underlying} {option_type.value} {int(strike)}",
                    underlying=underlying,
                    option_type=option_type,
                    strike_price=strike,
                    expiry_date=datetime.now() + timedelta(days=expiry_days),
                    market_price=max(1.0, abs(base_price - strike) + np.random.gamma(1, 10)),
                    bid_price=None,
                    ask_price=None,
                    volume=np.random.randint(50, 500),
                    open_interest=np.random.randint(20, 200),
                    implied_volatility=np.random.uniform(0.15, 0.35),
                    underlying_price=base_price
                )
                
                # 设置买卖价差
                spread = option.market_price * 0.015
                option.bid_price = max(0.1, option.market_price - spread/2)
                option.ask_price = option.market_price + spread/2
                
                options.append(option)
        
        return options
    
    @staticmethod
    def create_market_scenario(scenario_type: str = "normal") -> Dict[str, Any]:
        """创建市场场景"""
        scenarios = {
            "normal": {
                "volatility": 0.20,
                "drift": 0.0005,
                "jump_probability": 0.0,
                "correlation": 0.7
            },
            "high_volatility": {
                "volatility": 0.40,
                "drift": 0.0,
                "jump_probability": 0.05,
                "correlation": 0.5
            },
            "trending_up": {
                "volatility": 0.25,
                "drift": 0.002,
                "jump_probability": 0.01,
                "correlation": 0.8
            },
            "trending_down": {
                "volatility": 0.30,
                "drift": -0.0015,
                "jump_probability": 0.02,
                "correlation": 0.6
            }
        }
        
        return scenarios.get(scenario_type, scenarios["normal"])


@pytest.fixture
def test_data_factory():
    """测试数据工厂夹具"""
    return TestDataFactory()


# ==================== 性能测试支持 ====================

@pytest.fixture
def performance_timer():
    """性能计时器"""
    import time
    
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            self.end_time = time.perf_counter()
            return self.elapsed_time
        
        @property
        def elapsed_time(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return PerformanceTimer()


@pytest.fixture
def memory_monitor():
    """内存监控器"""
    import psutil
    import os
    
    class MemoryMonitor:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.initial_memory = None
            self.peak_memory = None
        
        def start(self):
            self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.initial_memory
        
        def update(self):
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
            return current_memory
        
        @property
        def memory_increase(self):
            if self.initial_memory:
                return self.peak_memory - self.initial_memory
            return None
    
    return MemoryMonitor()


# ==================== 清理夹具 ====================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """每个测试后的清理"""
    yield
    
    # 清理可能的异步任务
    try:
        loop = asyncio.get_event_loop()
        pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
        if pending_tasks:
            for task in pending_tasks:
                task.cancel()
    except RuntimeError:
        pass  # 没有运行中的事件循环
    
    # 强制垃圾回收
    import gc
    gc.collect()


# ==================== 测试标记辅助函数 ====================

def requires_real_data():
    """需要真实数据的测试标记"""
    return pytest.mark.skipif(
        not os.getenv('USE_REAL_DATA', False),
        reason="需要设置 USE_REAL_DATA=1 环境变量来运行真实数据测试"
    )


def slow_test():
    """慢速测试标记"""
    return pytest.mark.slow


def integration_test():
    """集成测试标记"""
    return pytest.mark.integration


def performance_test():
    """性能测试标记"""
    return pytest.mark.performance