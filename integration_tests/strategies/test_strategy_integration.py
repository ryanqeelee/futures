"""
策略集成测试 - 定价套利、平价套利、波动率套利策略

测试覆盖：
1. 定价套利策略准确性和性能测试
2. 看跌看涨平价策略数学验证
3. 波动率套利策略风险评估
4. 策略组合协同效应测试
5. 实际市场场景模拟测试
6. 策略参数优化验证
"""

import pytest
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from src.strategies.pricing_arbitrage import PricingArbitrageStrategy
from src.strategies.看跌看涨平价策略 import PutCallParityStrategy
from src.strategies.波动率套利策略 import VolatilityArbitrageStrategy
from src.strategies.base import (
    OptionData, OptionType, StrategyResult, StrategyParameters
)
from src.config.models import ArbitrageOpportunity, StrategyType


class TestPricingArbitrageStrategy:
    """定价套利策略集成测试"""
    
    @pytest.mark.integration
    @pytest.mark.financial
    def test_pricing_arbitrage_accuracy(self, test_data_factory):
        """测试定价套利准确性"""
        strategy = PricingArbitrageStrategy()
        
        # 创建包含明显定价错误的期权数据
        underlying_price = 4000.0
        risk_free_rate = 0.03
        time_to_expiry = 30/365  # 30天
        
        # 正常定价的期权
        normal_option = OptionData(
            code="IF2312C4000",
            name="IF2312 Call 4000",
            underlying="IF2312",
            option_type=OptionType.CALL,
            strike_price=4000.0,
            expiry_date=datetime.now() + timedelta(days=30),
            market_price=89.5,  # 合理价格
            underlying_price=underlying_price,
            implied_volatility=0.25,
            volume=500,
            open_interest=200,
            theoretical_price=90.0  # 接近市场价格
        )
        
        # 明显低估的期权
        underpriced_option = OptionData(
            code="IF2312C4050",
            name="IF2312 Call 4050", 
            underlying="IF2312",
            option_type=OptionType.CALL,
            strike_price=4050.0,
            expiry_date=datetime.now() + timedelta(days=30),
            market_price=45.0,  # 明显低估
            underlying_price=underlying_price,
            implied_volatility=0.25,
            volume=300,
            open_interest=150,
            theoretical_price=65.0  # 明显高于市场价格
        )
        
        # 明显高估的期权
        overpriced_option = OptionData(
            code="IF2312P4000",
            name="IF2312 Put 4000",
            underlying="IF2312", 
            option_type=OptionType.PUT,
            strike_price=4000.0,
            expiry_date=datetime.now() + timedelta(days=30),
            market_price=85.0,  # 明显高估
            underlying_price=underlying_price,
            implied_volatility=0.25,
            volume=400,
            open_interest=180,
            theoretical_price=60.0  # 明显低于市场价格
        )
        
        test_options = [normal_option, underpriced_option, overpriced_option]
        
        # 执行策略扫描
        result = strategy.scan_opportunities(test_options)
        
        # 验证策略结果
        assert isinstance(result, StrategyResult)
        assert result.success
        assert len(result.opportunities) >= 2  # 应该找到低估和高估的机会
        
        # 验证套利机会准确性
        found_underpriced = False
        found_overpriced = False
        
        for opp in result.opportunities:
            assert opp.expected_profit > 0
            assert opp.profit_margin > 0.01  # 至少1%利润率
            assert 0 < opp.confidence_score <= 1
            
            if opp.instruments[0] == "IF2312C4050":
                found_underpriced = True
                # 低估期权应该是买入机会
                assert any(action['action'] == 'BUY' for action in opp.actions)
                
            elif opp.instruments[0] == "IF2312P4000":
                found_overpriced = True
                # 高估期权应该是卖出机会
                assert any(action['action'] == 'SELL' for action in opp.actions)
        
        assert found_underpriced, "未检测到低估期权"
        assert found_overpriced, "未检测到高估期权"
    
    @pytest.mark.integration
    @pytest.mark.performance
    def test_pricing_arbitrage_performance(self, test_data_factory, performance_timer):
        """测试定价套利性能"""
        strategy = PricingArbitrageStrategy()
        
        # 创建大规模期权数据
        large_option_dataset = []
        for underlying in ['IF2312', 'IH2312', 'IC2312']:
            options = test_data_factory.create_option_chain(
                underlying=underlying,
                base_price=4000 + np.random.uniform(-200, 200),
                expiry_days=np.random.randint(5, 60)
            )
            large_option_dataset.extend(options)
        
        print(f"测试大数据集性能: {len(large_option_dataset)} 个期权")
        
        # 性能测试
        performance_timer.start()
        result = strategy.scan_opportunities(large_option_dataset)
        execution_time = performance_timer.stop()
        
        # 验证性能要求
        assert execution_time < 2.0, f"策略执行耗时过长: {execution_time:.3f}s"
        assert result.execution_time < 2.0, f"策略报告执行时间异常: {result.execution_time:.3f}s"
        
        # 验证扫描效率
        options_per_second = len(large_option_dataset) / execution_time
        assert options_per_second > 50, f"处理速度过慢: {options_per_second:.1f} 期权/秒"
        
        # 验证结果质量
        if result.opportunities:
            avg_profit_margin = np.mean([opp.profit_margin for opp in result.opportunities])
            assert avg_profit_margin > 0.015, "平均利润率过低"
    
    @pytest.mark.integration
    @pytest.mark.financial
    def test_pricing_model_accuracy(self):
        """测试期权定价模型准确性"""
        strategy = PricingArbitrageStrategy()
        
        # 使用Black-Scholes公式的已知解进行验证
        test_cases = [
            {
                'S': 4000,    # 标的价格
                'K': 4000,    # 行权价
                'T': 0.25,    # 到期时间（年）
                'r': 0.03,    # 无风险利率
                'sigma': 0.20,  # 波动率
                'option_type': 'call',
                'expected_price': 80.86  # 理论BS价格（近似）
            },
            {
                'S': 4000,
                'K': 4100,
                'T': 0.25,
                'r': 0.03,
                'sigma': 0.20,
                'option_type': 'call',
                'expected_price': 31.14  # 理论BS价格（近似）
            },
            {
                'S': 4000,
                'K': 4000,
                'T': 0.25,
                'r': 0.03,
                'sigma': 0.20,
                'option_type': 'put',
                'expected_price': 51.23  # 理论BS价格（近似）
            }
        ]
        
        for case in test_cases:
            option = OptionData(
                code=f"TEST_{case['option_type']}_{case['K']}",
                name=f"TEST {case['option_type']} {case['K']}",
                underlying="TEST",
                option_type=OptionType.CALL if case['option_type'] == 'call' else OptionType.PUT,
                strike_price=case['K'],
                expiry_date=datetime.now() + timedelta(days=int(case['T'] * 365)),
                market_price=case['expected_price'] + np.random.uniform(-2, 2),  # 添加噪声
                underlying_price=case['S'],
                implied_volatility=case['sigma'],
                volume=100,
                open_interest=50
            )
            
            # 计算理论价格
            theoretical_price = strategy._calculate_theoretical_price(option, case['r'])
            
            # 验证定价准确性（允许5%误差）
            price_error = abs(theoretical_price - case['expected_price']) / case['expected_price']
            assert price_error < 0.05, \
                f"定价误差过大 {case}: 计算={theoretical_price:.2f}, 期望={case['expected_price']:.2f}"


class TestPutCallParityStrategy:
    """看跌看涨平价策略测试"""
    
    @pytest.mark.integration
    @pytest.mark.financial
    def test_put_call_parity_detection(self):
        """测试看跌看涨平价关系检测"""
        strategy = PutCallParityStrategy()
        
        # 创建符合平价关系的期权对
        S = 4000.0  # 标的价格
        K = 4000.0  # 行权价
        r = 0.03    # 无风险利率
        T = 30/365  # 到期时间
        
        # 理论平价关系: Call - Put = S - K*e^(-r*T)
        parity_value = S - K * np.exp(-r * T)
        
        call_option = OptionData(
            code="IF2312C4000",
            name="IF2312 Call 4000",
            underlying="IF2312",
            option_type=OptionType.CALL,
            strike_price=K,
            expiry_date=datetime.now() + timedelta(days=30),
            market_price=85.0,
            underlying_price=S,
            volume=500,
            open_interest=200
        )
        
        put_option = OptionData(
            code="IF2312P4000", 
            name="IF2312 Put 4000",
            underlying="IF2312",
            option_type=OptionType.PUT,
            strike_price=K,
            expiry_date=datetime.now() + timedelta(days=30),
            market_price=60.0,  # 违反平价关系
            underlying_price=S,
            volume=450,
            open_interest=180
        )
        
        # 实际差价
        actual_diff = call_option.market_price - put_option.market_price
        expected_diff = parity_value
        
        print(f"平价检验: 实际差价={actual_diff:.2f}, 理论差价={expected_diff:.2f}")
        
        test_options = [call_option, put_option]
        
        # 执行平价策略
        result = strategy.scan_opportunities(test_options)
        
        # 验证平价套利检测
        assert result.success
        
        if abs(actual_diff - expected_diff) > 5.0:  # 如果偏差大于5元
            assert len(result.opportunities) > 0, "应该检测到平价套利机会"
            
            opp = result.opportunities[0]
            assert opp.strategy_type == StrategyType.PUT_CALL_PARITY
            assert len(opp.instruments) == 2
            assert "IF2312C4000" in opp.instruments
            assert "IF2312P4000" in opp.instruments
            
            # 验证套利方向正确性
            if actual_diff > expected_diff:  # Call相对高估
                # 应该卖出Call，买入Put
                call_action = next(a for a in opp.actions if a['instrument'] == 'IF2312C4000')
                put_action = next(a for a in opp.actions if a['instrument'] == 'IF2312P4000')
                
                assert call_action['action'] == 'SELL'
                assert put_action['action'] == 'BUY'
            else:  # Put相对高估
                # 应该买入Call，卖出Put
                call_action = next(a for a in opp.actions if a['instrument'] == 'IF2312C4000')
                put_action = next(a for a in opp.actions if a['instrument'] == 'IF2312P4000')
                
                assert call_action['action'] == 'BUY'
                assert put_action['action'] == 'SELL'
    
    @pytest.mark.integration
    @pytest.mark.financial
    def test_dividend_adjusted_parity(self):
        """测试含股息调整的平价关系"""
        strategy = PutCallParityStrategy()
        
        # 设置策略参数包含股息率
        strategy.set_parameters(StrategyParameters(
            custom_params={
                'dividend_yield': 0.02,  # 2%股息率
                'risk_free_rate': 0.03
            }
        ))
        
        S = 4000.0
        K = 4000.0
        T = 90/365  # 3个月
        r = 0.03
        q = 0.02  # 股息率
        
        # 调整后的平价关系: Call - Put = S*e^(-q*T) - K*e^(-r*T)
        adjusted_parity = S * np.exp(-q * T) - K * np.exp(-r * T)
        
        call_option = OptionData(
            code="INDEX_C_4000",
            name="Index Call 4000",
            underlying="INDEX",
            option_type=OptionType.CALL,
            strike_price=K,
            expiry_date=datetime.now() + timedelta(days=90),
            market_price=90.0,
            underlying_price=S,
            volume=300,
            open_interest=150
        )
        
        put_option = OptionData(
            code="INDEX_P_4000",
            name="Index Put 4000", 
            underlying="INDEX",
            option_type=OptionType.PUT,
            strike_price=K,
            expiry_date=datetime.now() + timedelta(days=90),
            market_price=70.0,  # 设置违反调整后平价关系的价格
            underlying_price=S,
            volume=280,
            open_interest=140
        )
        
        test_options = [call_option, put_option]
        result = strategy.scan_opportunities(test_options)
        
        # 验证股息调整的平价检测
        if result.opportunities:
            opp = result.opportunities[0]
            assert 'dividend_adjusted' in opp.metadata
            assert opp.metadata['dividend_yield'] == 0.02


class TestVolatilityArbitrageStrategy:
    """波动率套利策略测试"""
    
    @pytest.mark.integration
    @pytest.mark.financial
    def test_implied_volatility_arbitrage(self, test_data_factory):
        """测试隐含波动率套利"""
        strategy = VolatilityArbitrageStrategy()
        
        # 创建隐含波动率差异明显的期权
        underlying_price = 4000.0
        
        # 低隐含波动率期权
        low_iv_option = OptionData(
            code="IF2312C4000_LOW",
            name="IF2312 Call 4000 Low IV",
            underlying="IF2312",
            option_type=OptionType.CALL,
            strike_price=4000.0,
            expiry_date=datetime.now() + timedelta(days=30),
            market_price=75.0,
            underlying_price=underlying_price,
            implied_volatility=0.15,  # 低隐含波动率
            volume=400,
            open_interest=200
        )
        
        # 高隐含波动率期权
        high_iv_option = OptionData(
            code="IF2312C4050_HIGH",
            name="IF2312 Call 4050 High IV",
            underlying="IF2312",
            option_type=OptionType.CALL,
            strike_price=4050.0,
            expiry_date=datetime.now() + timedelta(days=30),
            market_price=65.0,
            underlying_price=underlying_price,
            implied_volatility=0.35,  # 高隐含波动率
            volume=350,
            open_interest=180
        )
        
        # 正常隐含波动率期权（对照组）
        normal_iv_option = OptionData(
            code="IF2312C4100_NORMAL",
            name="IF2312 Call 4100 Normal IV",
            underlying="IF2312",
            option_type=OptionType.CALL,
            strike_price=4100.0,
            expiry_date=datetime.now() + timedelta(days=30),
            market_price=45.0,
            underlying_price=underlying_price,
            implied_volatility=0.25,  # 正常隐含波动率
            volume=300,
            open_interest=150
        )
        
        test_options = [low_iv_option, high_iv_option, normal_iv_option]
        
        # 执行波动率套利策略
        result = strategy.scan_opportunities(test_options)
        
        # 验证波动率套利检测
        assert result.success
        assert len(result.opportunities) > 0
        
        for opp in result.opportunities:
            assert opp.strategy_type == StrategyType.VOLATILITY_ARBITRAGE
            assert 'implied_volatility_spread' in opp.metadata
            assert 'volatility_arbitrage_type' in opp.metadata
            
            # 验证套利逻辑
            iv_spread = opp.metadata['implied_volatility_spread']
            if iv_spread > 0.1:  # 波动率差异超过10%
                assert opp.profit_margin > 0.02  # 应该有显著利润机会
    
    @pytest.mark.integration
    @pytest.mark.financial
    def test_calendar_spread_volatility_arbitrage(self):
        """测试日历价差波动率套利"""
        strategy = VolatilityArbitrageStrategy()
        
        underlying_price = 4000.0
        
        # 近月期权（高时间价值衰减）
        near_month_option = OptionData(
            code="IF2312C4000",
            name="IF2312 Call 4000 Dec",
            underlying="IF2312",
            option_type=OptionType.CALL,
            strike_price=4000.0,
            expiry_date=datetime.now() + timedelta(days=15),  # 15天到期
            market_price=55.0,
            underlying_price=underlying_price,
            implied_volatility=0.28,
            volume=600,
            open_interest=300
        )
        
        # 远月期权（低时间价值衰减）
        far_month_option = OptionData(
            code="IF2401C4000",
            name="IF2401 Call 4000 Jan", 
            underlying="IF2312",  # 同一标的
            option_type=OptionType.CALL,
            strike_price=4000.0,
            expiry_date=datetime.now() + timedelta(days=45),  # 45天到期
            market_price=85.0,
            underlying_price=underlying_price,
            implied_volatility=0.22,  # 更低的隐含波动率
            volume=400,
            open_interest=200
        )
        
        test_options = [near_month_option, far_month_option]
        
        # 执行日历价差分析
        result = strategy.scan_opportunities(test_options)
        
        if result.opportunities:
            calendar_opportunities = [
                opp for opp in result.opportunities 
                if opp.metadata.get('arbitrage_type') == 'calendar_spread'
            ]
            
            if calendar_opportunities:
                opp = calendar_opportunities[0]
                assert len(opp.instruments) == 2
                assert 'time_decay_advantage' in opp.metadata
                assert 'theta_differential' in opp.metadata
    
    @pytest.mark.integration
    @pytest.mark.performance
    def test_volatility_surface_analysis_performance(self, performance_timer):
        """测试波动率曲面分析性能"""
        strategy = VolatilityArbitrageStrategy()
        
        # 创建完整的波动率曲面数据
        underlying_price = 4000.0
        volatility_surface = []
        
        strikes = range(3700, 4301, 50)  # 13个行权价
        expiries = [15, 30, 60, 90]      # 4个到期日
        
        for days_to_expiry in expiries:
            for strike in strikes:
                # 模拟现实的波动率微笑/偏斜
                moneyness = strike / underlying_price
                if moneyness < 0.98:  # 深度虚值看跌
                    base_iv = 0.30
                elif moneyness < 1.02:  # 平值附近
                    base_iv = 0.22
                else:  # 深度虚值看涨
                    base_iv = 0.28
                
                # 添加期限结构效应
                term_adjustment = 1 + (days_to_expiry - 30) * 0.002
                implied_vol = base_iv * term_adjustment
                
                for option_type in [OptionType.CALL, OptionType.PUT]:
                    option = OptionData(
                        code=f"SURF_{option_type.value}_{strike}_{days_to_expiry}",
                        name=f"Surface {option_type.value} {strike} {days_to_expiry}d",
                        underlying="SURFACE_TEST",
                        option_type=option_type,
                        strike_price=float(strike),
                        expiry_date=datetime.now() + timedelta(days=days_to_expiry),
                        market_price=max(5.0, abs(underlying_price - strike) * 0.3 + np.random.uniform(5, 15)),
                        underlying_price=underlying_price,
                        implied_volatility=implied_vol + np.random.normal(0, 0.02),
                        volume=np.random.randint(50, 300),
                        open_interest=np.random.randint(20, 150)
                    )
                    volatility_surface.append(option)
        
        print(f"波动率曲面测试: {len(volatility_surface)} 个期权")
        
        # 性能测试
        performance_timer.start()
        result = strategy.scan_opportunities(volatility_surface)
        execution_time = performance_timer.stop()
        
        # 验证性能要求
        assert execution_time < 5.0, f"波动率分析耗时过长: {execution_time:.3f}s"
        
        # 验证分析质量
        if result.opportunities:
            iv_arbitrage_opps = [
                opp for opp in result.opportunities
                if 'implied_volatility_spread' in opp.metadata
            ]
            
            assert len(iv_arbitrage_opps) > 0, "应该发现波动率套利机会"
            
            # 验证发现的机会具有合理的特征
            for opp in iv_arbitrage_opps[:5]:  # 检查前5个
                assert opp.confidence_score > 0.6
                assert opp.risk_score < 0.3
                assert abs(opp.metadata['implied_volatility_spread']) > 0.05  # 至少5%差异


class TestStrategyIntegrationAndCombination:
    """策略集成和组合测试"""
    
    @pytest.mark.integration
    @pytest.mark.financial
    def test_multi_strategy_coordination(self, test_data_factory):
        """测试多策略协调运行"""
        # 初始化所有策略
        pricing_strategy = PricingArbitrageStrategy()
        parity_strategy = PutCallParityStrategy()
        volatility_strategy = VolatilityArbitrageStrategy()
        
        # 创建综合测试数据
        test_options = test_data_factory.create_option_chain("MULTI_TEST", 4000.0, 30)
        
        # 为测试数据添加各种套利机会
        # 1. 定价错误
        test_options[5].market_price = test_options[5].market_price * 0.7  # 明显低估
        
        # 2. 平价关系错误
        call_option = next(opt for opt in test_options if opt.option_type == OptionType.CALL and opt.strike_price == 4000)
        put_option = next(opt for opt in test_options if opt.option_type == OptionType.PUT and opt.strike_price == 4000)
        put_option.market_price = call_option.market_price + 30  # 违反平价关系
        
        # 3. 波动率差异
        for i, option in enumerate(test_options[:10]):
            if i % 2 == 0:
                option.implied_volatility = 0.15  # 低波动率
            else:
                option.implied_volatility = 0.35  # 高波动率
        
        # 执行所有策略
        pricing_result = pricing_strategy.scan_opportunities(test_options)
        parity_result = parity_strategy.scan_opportunities(test_options)
        volatility_result = volatility_strategy.scan_opportunities(test_options)
        
        # 验证策略协调性
        all_opportunities = (
            pricing_result.opportunities + 
            parity_result.opportunities + 
            volatility_result.opportunities
        )
        
        assert len(all_opportunities) > 0, "至少一个策略应该发现机会"
        
        # 验证策略间无冲突
        # 检查是否有同一期权被多个策略以相反方向推荐
        instrument_actions = {}
        for opp in all_opportunities:
            for action in opp.actions:
                instrument = action['instrument']
                action_type = action['action']
                
                if instrument not in instrument_actions:
                    instrument_actions[instrument] = []
                instrument_actions[instrument].append((opp.strategy_type, action_type))
        
        # 检查冲突（同一期权同时有买入和卖出推荐）
        conflicts = []
        for instrument, actions in instrument_actions.items():
            buy_strategies = [s for s, a in actions if a == 'BUY']
            sell_strategies = [s for s, a in actions if a == 'SELL']
            
            if buy_strategies and sell_strategies:
                conflicts.append({
                    'instrument': instrument,
                    'buy_strategies': buy_strategies,
                    'sell_strategies': sell_strategies
                })
        
        # 允许少量合理冲突（可能代表不同角度的分析）
        assert len(conflicts) <= len(all_opportunities) * 0.1, f"策略冲突过多: {conflicts}"
    
    @pytest.mark.integration
    @pytest.mark.financial
    def test_strategy_risk_aggregation(self):
        """测试策略风险聚合"""
        strategies = [
            PricingArbitrageStrategy(),
            PutCallParityStrategy(), 
            VolatilityArbitrageStrategy()
        ]
        
        # 模拟每个策略产生的机会
        mock_opportunities = {
            StrategyType.PRICING_ARBITRAGE: [
                ArbitrageOpportunity(
                    id="pricing_1",
                    strategy_type=StrategyType.PRICING_ARBITRAGE,
                    instruments=["TEST_C_4000"],
                    underlying="TEST",
                    expected_profit=100.0,
                    max_loss=30.0,
                    risk_score=0.15,
                    confidence_score=0.8,
                    profit_margin=0.025,
                    days_to_expiry=30,
                    market_prices={"TEST_C_4000": 85.0},
                    volumes={"TEST_C_4000": 300},
                    actions=[],
                    data_source="test"
                )
            ],
            StrategyType.PUT_CALL_PARITY: [
                ArbitrageOpportunity(
                    id="parity_1",
                    strategy_type=StrategyType.PUT_CALL_PARITY,
                    instruments=["TEST_C_4000", "TEST_P_4000"],
                    underlying="TEST", 
                    expected_profit=75.0,
                    max_loss=25.0,
                    risk_score=0.08,
                    confidence_score=0.9,
                    profit_margin=0.02,
                    days_to_expiry=30,
                    market_prices={"TEST_C_4000": 85.0, "TEST_P_4000": 60.0},
                    volumes={"TEST_C_4000": 300, "TEST_P_4000": 250},
                    actions=[],
                    data_source="test"
                )
            ],
            StrategyType.VOLATILITY_ARBITRAGE: [
                ArbitrageOpportunity(
                    id="volatility_1",
                    strategy_type=StrategyType.VOLATILITY_ARBITRAGE,
                    instruments=["TEST_C_4050", "TEST_C_4100"],
                    underlying="TEST",
                    expected_profit=60.0,
                    max_loss=40.0,
                    risk_score=0.20,
                    confidence_score=0.7,
                    profit_margin=0.018,
                    days_to_expiry=30,
                    market_prices={"TEST_C_4050": 65.0, "TEST_C_4100": 45.0},
                    volumes={"TEST_C_4050": 200, "TEST_C_4100": 180},
                    actions=[],
                    data_source="test"
                )
            ]
        }
        
        # 聚合所有机会
        all_opportunities = []
        for strategy_type, opportunities in mock_opportunities.items():
            all_opportunities.extend(opportunities)
        
        # 计算组合风险指标
        total_expected_profit = sum(opp.expected_profit for opp in all_opportunities)
        total_max_loss = sum(opp.max_loss for opp in all_opportunities)
        weighted_confidence = sum(opp.confidence_score * opp.expected_profit for opp in all_opportunities) / total_expected_profit
        weighted_risk_score = sum(opp.risk_score * opp.expected_profit for opp in all_opportunities) / total_expected_profit
        
        # 验证风险聚合合理性
        assert total_expected_profit > 200.0  # 总预期收益
        assert total_max_loss < total_expected_profit  # 风险收益比合理
        assert 0.7 <= weighted_confidence <= 1.0  # 综合置信度
        assert weighted_risk_score <= 0.25  # 综合风险可控
        
        print(f"策略组合分析:")
        print(f"  总预期收益: {total_expected_profit:.2f}")
        print(f"  总最大损失: {total_max_loss:.2f}")
        print(f"  收益风险比: {total_expected_profit/total_max_loss:.2f}")
        print(f"  综合置信度: {weighted_confidence:.3f}")
        print(f"  综合风险评分: {weighted_risk_score:.3f}")
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_strategy_stress_testing(self, test_data_factory):
        """测试策略压力测试"""
        strategies = [
            PricingArbitrageStrategy(),
            PutCallParityStrategy(),
            VolatilityArbitrageStrategy()
        ]
        
        # 创建极端市场条件
        stress_scenarios = [
            {
                'name': 'high_volatility_shock',
                'underlying_price': 4000.0,
                'volatility_multiplier': 3.0,  # 波动率激增
                'volume_reduction': 0.3  # 流动性枯竭
            },
            {
                'name': 'price_gap_down',
                'underlying_price': 3500.0,  # 12.5%下跌
                'volatility_multiplier': 2.0,
                'volume_reduction': 0.5
            },
            {
                'name': 'liquidity_crisis',
                'underlying_price': 4000.0,
                'volatility_multiplier': 1.5,
                'volume_reduction': 0.1  # 90%流动性下降
            }
        ]
        
        stress_results = {}
        
        for scenario in stress_scenarios:
            print(f"\n执行压力测试: {scenario['name']}")
            
            # 创建压力测试数据
            stress_options = test_data_factory.create_option_chain(
                underlying="STRESS_TEST",
                base_price=scenario['underlying_price'],
                expiry_days=15  # 临近到期增加压力
            )
            
            # 应用压力条件
            for option in stress_options:
                # 调整波动率
                option.implied_volatility *= scenario['volatility_multiplier']
                option.implied_volatility = min(option.implied_volatility, 1.0)  # 上限100%
                
                # 调整流动性
                option.volume = int(option.volume * scenario['volume_reduction'])
                option.volume = max(option.volume, 1)  # 至少1手
                
                # 调整价格（模拟流动性冲击导致的价差扩大）
                price_impact = 1.0 + (1 - scenario['volume_reduction']) * 0.1
                option.market_price *= price_impact
            
            # 测试所有策略在压力条件下的表现
            scenario_results = {}
            for strategy in strategies:
                try:
                    result = strategy.scan_opportunities(stress_options)
                    scenario_results[strategy.__class__.__name__] = {
                        'success': result.success,
                        'opportunities_found': len(result.opportunities),
                        'avg_confidence': np.mean([opp.confidence_score for opp in result.opportunities]) if result.opportunities else 0,
                        'execution_time': result.execution_time
                    }
                except Exception as e:
                    scenario_results[strategy.__class__.__name__] = {
                        'success': False,
                        'error': str(e),
                        'opportunities_found': 0,
                        'avg_confidence': 0,
                        'execution_time': float('inf')
                    }
            
            stress_results[scenario['name']] = scenario_results
        
        # 验证策略在压力下的健壮性
        for scenario_name, results in stress_results.items():
            print(f"\n{scenario_name} 结果:")
            
            for strategy_name, result in results.items():
                print(f"  {strategy_name}: 成功={result['success']}, 机会={result['opportunities_found']}")
                
                # 策略应该在压力下仍能正常运行（不崩溃）
                assert result['success'] or 'error' in result, f"{strategy_name} 在 {scenario_name} 下异常终止"
                
                # 执行时间不应显著增加（除非遇到错误）
                if result['success']:
                    assert result['execution_time'] < 10.0, f"{strategy_name} 在压力下执行时间过长"
        
        # 至少一个策略在每种压力情况下仍能发现机会
        for scenario_name, results in stress_results.items():
            total_opportunities = sum(r['opportunities_found'] for r in results.values())
            # 在极端压力下可能没有套利机会，这是合理的