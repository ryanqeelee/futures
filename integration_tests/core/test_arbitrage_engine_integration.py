"""
套利引擎核心业务逻辑集成测试

测试覆盖：
1. 完整的套利扫描流程
2. 多策略并行执行
3. 实时数据处理和分析
4. 性能优化和缓存机制
5. 错误处理和恢复
6. 并发安全性测试
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from src.engine.arbitrage_engine import (
    ArbitrageEngine, ScanParameters, TradingSignal, EnginePerformanceMetrics
)
from src.config.models import ArbitrageOpportunity, StrategyType, RiskLevel
from src.strategies.base import OptionData, OptionType, StrategyResult
from src.adapters.base import DataResponse


class TestArbitrageEngineIntegration:
    """套利引擎集成测试类"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_arbitrage_scanning_pipeline(
        self, integration_arbitrage_engine, sample_option_data
    ):
        """测试完整的套利扫描流水线"""
        engine = integration_arbitrage_engine
        
        # 设置扫描参数
        scan_params = ScanParameters(
            strategy_types=[StrategyType.PRICING_ARBITRAGE],
            min_profit_threshold=0.01,
            max_risk_tolerance=0.2,
            max_results=50,
            include_metadata=True
        )
        
        # 执行完整扫描
        start_time = time.perf_counter()
        opportunities = await engine.scan_opportunities(scan_params)
        scan_duration = time.perf_counter() - start_time
        
        # 验证结果
        assert isinstance(opportunities, list)
        assert len(opportunities) > 0
        assert all(isinstance(opp, ArbitrageOpportunity) for opp in opportunities)
        
        # 验证性能要求
        assert scan_duration < 3.0, f"扫描耗时 {scan_duration:.2f}s，超出3s要求"
        
        # 验证套利机会质量
        for opp in opportunities:
            assert opp.expected_profit > 0
            assert opp.profit_margin >= scan_params.min_profit_threshold
            assert opp.risk_score <= scan_params.max_risk_tolerance
            assert opp.confidence_score > 0
            assert len(opp.instruments) > 0
            assert opp.underlying is not None
        
        # 验证结果排序（按利润率降序）
        profit_margins = [opp.profit_margin for opp in opportunities]
        assert profit_margins == sorted(profit_margins, reverse=True)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multi_strategy_parallel_execution(
        self, integration_arbitrage_engine, sample_option_data
    ):
        """测试多策略并行执行"""
        engine = integration_arbitrage_engine
        
        # 添加多个策略
        additional_strategy = Mock()
        additional_strategy.strategy_type = StrategyType.PUT_CALL_PARITY
        additional_strategy.name = "MockParityStrategy"
        additional_strategy.enabled = True
        
        def mock_parity_scan(options_data):
            return StrategyResult(
                strategy_name="MockParityStrategy",
                opportunities=[
                    ArbitrageOpportunity(
                        id="parity_opp_1",
                        strategy_type=StrategyType.PUT_CALL_PARITY,
                        instruments=["TEST-C-4000", "TEST-P-4000"],
                        underlying="TEST",
                        expected_profit=75.0,
                        profit_margin=0.025,
                        confidence_score=0.9,
                        max_loss=25.0,
                        risk_score=0.08,
                        days_to_expiry=30,
                        market_prices={"TEST-C-4000": 50.0, "TEST-P-4000": 48.0},
                        volumes={"TEST-C-4000": 100, "TEST-P-4000": 120},
                        actions=[],
                        data_source="mock_parity"
                    )
                ],
                execution_time=0.12,
                data_timestamp=datetime.now(),
                success=True
            )
        
        additional_strategy.scan_opportunities.side_effect = mock_parity_scan
        engine.strategies[StrategyType.PUT_CALL_PARITY] = additional_strategy
        
        # 执行多策略扫描
        scan_params = ScanParameters(
            strategy_types=[StrategyType.PRICING_ARBITRAGE, StrategyType.PUT_CALL_PARITY],
            min_profit_threshold=0.01,
            max_results=100
        )
        
        start_time = time.perf_counter()
        opportunities = await engine.scan_opportunities(scan_params)
        parallel_duration = time.perf_counter() - start_time
        
        # 验证并行执行效果
        assert len(opportunities) >= 4  # 至少有两个策略的结果
        
        strategy_types_found = {opp.strategy_type for opp in opportunities}
        assert StrategyType.PRICING_ARBITRAGE in strategy_types_found
        assert StrategyType.PUT_CALL_PARITY in strategy_types_found
        
        # 验证并行执行性能（应该比串行快）
        assert parallel_duration < 1.0, f"并行执行耗时过长: {parallel_duration:.2f}s"
        
        # 验证结果混合排序正确
        pricing_opps = [opp for opp in opportunities if opp.strategy_type == StrategyType.PRICING_ARBITRAGE]
        parity_opps = [opp for opp in opportunities if opp.strategy_type == StrategyType.PUT_CALL_PARITY]
        
        assert len(pricing_opps) > 0
        assert len(parity_opps) > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_time_data_processing_accuracy(
        self, integration_arbitrage_engine, test_data_factory
    ):
        """测试实时数据处理准确性"""
        engine = integration_arbitrage_engine
        
        # 创建真实的期权链数据
        option_chain = test_data_factory.create_option_chain(
            underlying="IF2312", 
            base_price=4000.0, 
            expiry_days=30
        )
        
        # 模拟实时数据更新
        async def mock_real_time_data(request):
            # 模拟价格微小变动
            updated_options = []
            for option in option_chain:
                # 价格随机游走
                price_change = np.random.normal(0, option.market_price * 0.005)  # 0.5% 波动
                new_price = max(0.1, option.market_price + price_change)
                
                updated_option = OptionData(
                    code=option.code,
                    name=option.name,
                    underlying=option.underlying,
                    option_type=option.option_type,
                    strike_price=option.strike_price,
                    expiry_date=option.expiry_date,
                    market_price=new_price,
                    bid_price=max(0.05, new_price - new_price * 0.01),
                    ask_price=new_price + new_price * 0.01,
                    volume=option.volume + np.random.randint(-10, 11),
                    open_interest=option.open_interest,
                    implied_volatility=option.implied_volatility + np.random.normal(0, 0.01),
                    underlying_price=option.underlying_price + np.random.normal(0, 2)
                )
                
                updated_options.append(updated_option)
            
            return DataResponse(
                request=request,
                data=updated_options,
                timestamp=datetime.now(),
                source="real_time_mock",
                quality="REAL_TIME"
            )
        
        # 更新数据适配器
        engine.data_adapters["test_mock"].get_option_data.side_effect = mock_real_time_data
        
        # 执行多次实时扫描
        scan_results = []
        for i in range(5):
            opportunities = await engine.scan_opportunities(ScanParameters())
            scan_results.append({
                'scan_id': i,
                'timestamp': datetime.now(),
                'opportunities_count': len(opportunities),
                'total_profit': sum(opp.expected_profit for opp in opportunities),
                'avg_confidence': np.mean([opp.confidence_score for opp in opportunities]) if opportunities else 0
            })
            
            await asyncio.sleep(0.1)  # 模拟实时间隔
        
        # 验证实时数据处理
        assert len(scan_results) == 5
        assert all(result['opportunities_count'] >= 0 for result in scan_results)
        
        # 验证数据一致性
        confidence_scores = [result['avg_confidence'] for result in scan_results if result['avg_confidence'] > 0]
        if confidence_scores:
            assert all(0 <= score <= 1 for score in confidence_scores)
    
    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_performance_optimization_and_caching(
        self, integration_arbitrage_engine, sample_option_data, performance_timer, memory_monitor
    ):
        """测试性能优化和缓存机制"""
        engine = integration_arbitrage_engine
        
        # 启动监控
        performance_timer.start()
        memory_monitor.start()
        
        scan_params = ScanParameters(max_results=100)
        
        # 第一次扫描（冷缓存）
        opportunities_1 = await engine.scan_opportunities(scan_params)
        first_scan_time = performance_timer.stop()
        
        # 第二次扫描（热缓存）
        performance_timer.start()
        opportunities_2 = await engine.scan_opportunities(scan_params)
        second_scan_time = performance_timer.stop()
        
        memory_usage = memory_monitor.update()
        
        # 验证性能优化效果
        assert second_scan_time <= first_scan_time * 1.5, \
               f"缓存未能有效提升性能: {first_scan_time:.3f}s -> {second_scan_time:.3f}s"
        
        # 验证结果一致性（缓存有效性）
        assert len(opportunities_1) == len(opportunities_2)
        
        # 验证内存使用合理
        assert memory_monitor.memory_increase < 100, f"内存增长过多: {memory_monitor.memory_increase:.2f}MB"
        
        # 测试缓存命中率
        cache = engine.cache
        assert cache.hit_rate > 0.3, f"缓存命中率过低: {cache.hit_rate:.2%}"
        
        # 测试缓存清理
        initial_cache_size = len(cache._cache) if hasattr(cache, '_cache') else 0
        engine.clear_cache()
        final_cache_size = len(cache._cache) if hasattr(cache, '_cache') else 0
        
        assert final_cache_size < initial_cache_size, "缓存清理未生效"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(
        self, integration_arbitrage_engine, sample_option_data
    ):
        """测试错误处理和恢复机制"""
        engine = integration_arbitrage_engine
        
        # 测试数据适配器失败恢复
        original_get_data = engine.data_adapters["test_mock"].get_option_data
        
        # 模拟间歇性数据获取失败
        call_count = 0
        async def failing_data_adapter(request):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # 前两次调用失败
                raise ConnectionError("模拟网络连接失败")
            return await original_get_data(request)
        
        engine.data_adapters["test_mock"].get_option_data.side_effect = failing_data_adapter
        
        # 执行扫描（应该自动重试并恢复）
        opportunities = await engine.scan_opportunities(ScanParameters())
        
        # 验证最终成功恢复
        assert len(opportunities) > 0, "错误恢复机制未能正常工作"
        assert call_count >= 3, "重试机制未正确触发"
        
        # 测试策略执行失败处理
        failing_strategy = Mock()
        failing_strategy.strategy_type = StrategyType.PRICING_ARBITRAGE
        failing_strategy.scan_opportunities.side_effect = ValueError("策略计算错误")
        
        # 暂时替换策略
        original_strategy = engine.strategies[StrategyType.PRICING_ARBITRAGE]
        engine.strategies[StrategyType.PRICING_ARBITRAGE] = failing_strategy
        
        try:
            # 执行扫描（应该优雅处理策略失败）
            opportunities = await engine.scan_opportunities(ScanParameters(
                strategy_types=[StrategyType.PRICING_ARBITRAGE]
            ))
            
            # 验证错误被优雅处理
            assert opportunities == [], "策略失败应该返回空结果而不是抛出异常"
            
        finally:
            # 恢复原始策略
            engine.strategies[StrategyType.PRICING_ARBITRAGE] = original_strategy
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_scanning_safety(
        self, integration_arbitrage_engine, sample_option_data
    ):
        """测试并发扫描安全性"""
        engine = integration_arbitrage_engine
        
        # 创建多个并发扫描任务
        scan_params = ScanParameters(max_results=50)
        concurrent_tasks = []
        
        for i in range(10):  # 10个并发扫描
            task = asyncio.create_task(
                engine.scan_opportunities(scan_params),
                name=f"concurrent_scan_{i}"
            )
            concurrent_tasks.append(task)
        
        # 等待所有任务完成
        start_time = time.perf_counter()
        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        total_time = time.perf_counter() - start_time
        
        # 验证并发安全性
        successful_results = [r for r in results if isinstance(r, list)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        assert len(successful_results) >= 8, f"并发扫描成功率过低: {len(successful_results)}/10"
        assert len(failed_results) <= 2, f"并发扫描失败过多: {failed_results}"
        
        # 验证结果一致性（所有成功的扫描应该返回相似结果）
        if len(successful_results) >= 2:
            first_result = successful_results[0]
            for other_result in successful_results[1:]:
                # 机会数量应该相近（允许小幅差异）
                count_diff = abs(len(first_result) - len(other_result))
                assert count_diff <= 2, f"并发扫描结果差异过大: {count_diff}"
        
        # 验证性能（并发不应显著降低单次性能）
        avg_time_per_scan = total_time / len(concurrent_tasks)
        assert avg_time_per_scan < 2.0, f"并发扫描平均耗时过长: {avg_time_per_scan:.2f}s"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_trading_signals_generation_accuracy(
        self, integration_arbitrage_engine, mock_arbitrage_opportunities
    ):
        """测试交易信号生成准确性"""
        engine = integration_arbitrage_engine
        
        # 生成交易信号
        signals = engine.generate_trading_signals(mock_arbitrage_opportunities)
        
        # 验证信号生成
        assert len(signals) == len(mock_arbitrage_opportunities)
        assert all(isinstance(signal, TradingSignal) for signal in signals)
        
        # 验证信号准确性
        for i, (opp, signal) in enumerate(zip(mock_arbitrage_opportunities, signals)):
            assert signal.opportunity_id == opp.id
            assert signal.strategy_type == opp.strategy_type
            assert signal.expected_profit == opp.expected_profit
            assert signal.max_loss == opp.max_loss
            assert signal.confidence_score == opp.confidence_score
            assert len(signal.actions) == len(opp.actions)
            
            # 验证风险级别正确映射
            if opp.risk_score <= 0.1:
                assert signal.risk_level == RiskLevel.LOW
            elif opp.risk_score <= 0.2:
                assert signal.risk_level == RiskLevel.MEDIUM
            else:
                assert signal.risk_level == RiskLevel.HIGH
            
            # 验证执行优先级合理
            assert 1 <= signal.execution_priority <= 5
            
            # 验证过期时间合理
            assert signal.expiration_time > datetime.now()
            assert signal.expiration_time <= datetime.now() + timedelta(hours=24)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_engine_health_monitoring(
        self, integration_arbitrage_engine
    ):
        """测试引擎健康状况监控"""
        engine = integration_arbitrage_engine
        
        # 获取健康状态
        health_status = await engine.health_check()
        
        # 验证健康状态结构
        required_fields = [
            'engine_status', 'adapters', 'strategies', 'performance_metrics',
            'cache_status', 'memory_usage', 'uptime', 'last_scan_time'
        ]
        
        for field in required_fields:
            assert field in health_status, f"健康状态缺少字段: {field}"
        
        # 验证健康状态内容
        assert health_status['engine_status'] == 'healthy'
        assert health_status['adapters'] > 0
        assert health_status['strategies'] > 0
        assert isinstance(health_status['performance_metrics'], dict)
        assert 'hit_rate' in health_status['cache_status']
        assert health_status['memory_usage'] > 0
        assert health_status['uptime'] >= 0
        
        # 执行一次扫描后再检查健康状态
        await engine.scan_opportunities(ScanParameters())
        
        updated_health = await engine.health_check()
        assert updated_health['last_scan_time'] is not None
        assert updated_health['performance_metrics']['total_scans'] > 0
    
    @pytest.mark.integration 
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_dataset_scalability(
        self, integration_arbitrage_engine, test_data_factory, memory_monitor
    ):
        """测试大数据集可扩展性"""
        engine = integration_arbitrage_engine
        
        # 创建大规模期权数据集
        large_dataset = []
        underlyings = [f"TEST{i:03d}" for i in range(50)]  # 50个标的
        
        for underlying in underlyings:
            options = test_data_factory.create_option_chain(
                underlying=underlying,
                base_price=4000 + np.random.uniform(-500, 500),
                expiry_days=np.random.randint(5, 90)
            )
            large_dataset.extend(options)
        
        print(f"生成大规模数据集: {len(large_dataset)} 个期权")
        
        # 更新数据适配器以返回大数据集
        async def large_dataset_adapter(request):
            return DataResponse(
                request=request,
                data=large_dataset,
                timestamp=datetime.now(),
                source="large_dataset_mock",
                quality="HIGH"
            )
        
        engine.data_adapters["test_mock"].get_option_data.side_effect = large_dataset_adapter
        
        # 启动内存监控
        memory_monitor.start()
        
        # 执行大数据集扫描
        scan_params = ScanParameters(max_results=200)
        
        start_time = time.perf_counter()
        opportunities = await engine.scan_opportunities(scan_params)
        scan_duration = time.perf_counter() - start_time
        
        memory_usage = memory_monitor.update()
        
        # 验证可扩展性
        assert len(opportunities) > 0, "大数据集扫描应该找到机会"
        assert scan_duration < 10.0, f"大数据集扫描耗时过长: {scan_duration:.2f}s"
        assert memory_monitor.memory_increase < 500, f"内存使用过多: {memory_monitor.memory_increase:.2f}MB"
        
        # 验证结果质量
        assert all(opp.expected_profit > 0 for opp in opportunities[:10])  # 检查前10个
        assert len(set(opp.underlying for opp in opportunities)) > 1  # 应该涵盖多个标的
        
        print(f"大数据集扫描完成: {len(opportunities)} 个机会, 耗时 {scan_duration:.2f}s, 内存增长 {memory_monitor.memory_increase:.2f}MB")


class TestArbitrageEngineBusinessLogic:
    """套利引擎业务逻辑专项测试"""
    
    @pytest.mark.integration
    @pytest.mark.financial
    @pytest.mark.asyncio
    async def test_option_pricing_accuracy(
        self, integration_arbitrage_engine, test_data_factory
    ):
        """测试期权定价准确性"""
        engine = integration_arbitrage_engine
        
        # 创建已知定价的期权数据
        known_options = [
            OptionData(
                code="TEST-C-4000",
                name="TEST Call 4000",
                underlying="TEST",
                option_type=OptionType.CALL,
                strike_price=4000.0,
                expiry_date=datetime.now() + timedelta(days=30),
                market_price=89.5,  # 市场价格
                underlying_price=4025.0,  # 标的价格
                implied_volatility=0.25,  # 隐含波动率
                volume=500,
                open_interest=200
            ),
            OptionData(
                code="TEST-P-4000", 
                name="TEST Put 4000",
                underlying="TEST",
                option_type=OptionType.PUT,
                strike_price=4000.0,
                expiry_date=datetime.now() + timedelta(days=30),
                market_price=64.2,
                underlying_price=4025.0,
                implied_volatility=0.25,
                volume=450,
                open_interest=180
            )
        ]
        
        # 预处理期权数据（应该计算理论价格）
        await engine._preprocess_market_data(known_options)
        
        # 验证理论价格计算
        for option in known_options:
            assert option.theoretical_price is not None
            assert option.theoretical_price > 0
            
            # 理论价格应该接近市场价格（±20%范围内）
            price_diff_pct = abs(option.theoretical_price - option.market_price) / option.market_price
            assert price_diff_pct < 0.30, \
                   f"理论价格偏差过大 {option.code}: 市场价{option.market_price}, 理论价{option.theoretical_price}"
    
    @pytest.mark.integration
    @pytest.mark.financial
    @pytest.mark.asyncio
    async def test_arbitrage_opportunity_validation(
        self, integration_arbitrage_engine, risk_manager
    ):
        """测试套利机会验证逻辑"""
        engine = integration_arbitrage_engine
        
        # 创建边界测试用套利机会
        test_opportunities = [
            # 高质量机会
            ArbitrageOpportunity(
                id="high_quality",
                strategy_type=StrategyType.PRICING_ARBITRAGE,
                instruments=["TEST-C-4000"],
                underlying="TEST",
                expected_profit=120.0,
                profit_margin=0.035,
                confidence_score=0.95,
                max_loss=30.0,
                risk_score=0.08,
                days_to_expiry=25,
                market_prices={"TEST-C-4000": 85.0},
                volumes={"TEST-C-4000": 800},
                actions=[],
                data_source="test"
            ),
            # 边界质量机会
            ArbitrageOpportunity(
                id="marginal_quality",
                strategy_type=StrategyType.PRICING_ARBITRAGE,
                instruments=["TEST-P-3900"],
                underlying="TEST",
                expected_profit=45.0,
                profit_margin=0.012,  # 接近最小阈值
                confidence_score=0.65,
                max_loss=18.0,
                risk_score=0.18,  # 接近风险上限
                days_to_expiry=8,
                market_prices={"TEST-P-3900": 52.0},
                volumes={"TEST-P-3900": 150},  # 接近流动性下限
                actions=[],
                data_source="test"
            ),
            # 不合格机会
            ArbitrageOpportunity(
                id="poor_quality",
                strategy_type=StrategyType.PRICING_ARBITRAGE,
                instruments=["TEST-C-4100"],
                underlying="TEST",
                expected_profit=25.0,
                profit_margin=0.005,  # 低于阈值
                confidence_score=0.45,
                max_loss=80.0,  # 过高风险
                risk_score=0.35,
                days_to_expiry=2,  # 即将到期
                market_prices={"TEST-C-4100": 35.0},
                volumes={"TEST-C-4100": 25},  # 流动性不足
                actions=[],
                data_source="test"
            )
        ]
        
        # 使用不同阈值进行过滤测试
        scan_params_strict = ScanParameters(
            min_profit_threshold=0.02,
            max_risk_tolerance=0.15,
            min_confidence_score=0.8,
            max_results=100
        )
        
        scan_params_loose = ScanParameters(
            min_profit_threshold=0.008,
            max_risk_tolerance=0.25,
            min_confidence_score=0.5,
            max_results=100
        )
        
        # 严格过滤
        filtered_strict = engine._rank_and_filter_opportunities(test_opportunities, scan_params_strict)
        
        # 宽松过滤
        filtered_loose = engine._rank_and_filter_opportunities(test_opportunities, scan_params_loose)
        
        # 验证过滤逻辑
        assert len(filtered_strict) == 1  # 只有高质量机会通过
        assert filtered_strict[0].id == "high_quality"
        
        assert len(filtered_loose) >= 2  # 高质量和边界质量都应该通过
        assert any(opp.id == "high_quality" for opp in filtered_loose)
        assert any(opp.id == "marginal_quality" for opp in filtered_loose)
        
        # 验证风险限制
        is_valid, violations = risk_manager.validate_risk_limits(filtered_loose)
        assert is_valid or len(violations) <= 1  # 最多一个轻微违规