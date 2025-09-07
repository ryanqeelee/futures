"""
基于价格预测的期权交易策略

该策略通过预测期货价格走势，选择合适的期权组合进行低风险交易。
核心思路：
1. 使用机器学习模型预测期货价格方向和幅度
2. 根据预测结果和置信度选择期权交易策略
3. 通过希腊字母管理风险，实现低风险高收益
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging
from enum import Enum

from ..config.models import StrategyType, ArbitrageOpportunity
from .base import (
    BaseStrategy, StrategyResult, StrategyParameters, OptionData, 
    TradingAction, ActionType, RiskMetrics, RiskLevel, StrategyRegistry,
    OptionType
)


class PredictionSignal(str, Enum):
    """预测信号类型"""
    STRONG_BULLISH = "strong_bullish"    # 强烈看涨
    BULLISH = "bullish"                  # 看涨
    NEUTRAL = "neutral"                  # 中性
    BEARISH = "bearish"                  # 看跌
    STRONG_BEARISH = "strong_bearish"    # 强烈看跌


class OptionStrategy(str, Enum):
    """期权交易策略类型"""
    LONG_CALL = "long_call"              # 买入看涨期权
    LONG_PUT = "long_put"                # 买入看跌期权
    BULL_CALL_SPREAD = "bull_call_spread"  # 牛市看涨价差
    BEAR_PUT_SPREAD = "bear_put_spread"    # 熊市看跌价差
    STRADDLE = "straddle"                # 跨式组合
    IRON_CONDOR = "iron_condor"          # 铁鹰价差
    COLLAR = "collar"                    # 领口策略


@dataclass
class PredictionResult:
    """价格预测结果"""
    signal: PredictionSignal
    confidence: float  # 置信度 0-1
    predicted_price: float  # 预测价格
    current_price: float  # 当前价格
    time_horizon: int  # 预测时间范围（天）
    expected_move: float  # 预期涨跌幅
    volatility_forecast: float  # 波动率预测


@dataclass
class StrategyRecommendation:
    """策略推荐结果"""
    strategy: OptionStrategy
    options: List[OptionData]  # 涉及的期权
    actions: List[TradingAction]  # 交易动作
    expected_profit: float  # 预期收益
    max_loss: float  # 最大损失
    breakeven_points: List[float]  # 盈亏平衡点
    greeks: Dict[str, float]  # 希腊字母敞口


class PredictionBasedParameters(StrategyParameters):
    """基于预测的期权策略参数"""
    
    # 预测相关参数
    min_prediction_confidence: float = 0.65  # 最小预测置信度
    max_time_to_expiry: int = 60  # 最大到期时间（天）
    min_time_to_expiry: int = 7   # 最小到期时间（天）
    
    # 风险管理参数
    max_single_position_risk: float = 0.03  # 单笔最大风险（占总资金比例）
    max_delta_exposure: float = 0.3  # 最大Delta敞口
    max_vega_exposure: float = 0.2   # 最大Vega敞口
    max_theta_exposure: float = -0.1  # 最大Theta敞口
    
    # 期权选择参数
    preferred_moneyness_range: Tuple[float, float] = (0.9, 1.1)  # 行权价偏好范围
    min_open_interest: int = 100  # 最小持仓量
    max_bid_ask_spread: float = 0.05  # 最大买卖价差比例
    
    # 策略选择参数
    enable_directional_strategies: bool = True  # 启用方向性策略
    enable_volatility_strategies: bool = True   # 启用波动率策略
    enable_spread_strategies: bool = True       # 启用价差策略
    
    # 模型参数
    prediction_features: List[str] = None  # 预测特征列表
    model_retrain_days: int = 30  # 模型重训练间隔（天）
    
    def __post_init__(self):
        if self.prediction_features is None:
            self.prediction_features = [
                'sma_5', 'sma_10', 'sma_20',  # 移动平均
                'rsi_14', 'macd', 'bb_upper', 'bb_lower',  # 技术指标
                'volume_sma', 'volatility_20',  # 成交量和波动率
                'momentum_5', 'momentum_10'  # 价格动量
            ]


@StrategyRegistry.register(StrategyType.PREDICTION_BASED)
class PredictionBasedStrategy(BaseStrategy):
    """
    基于价格预测的期权交易策略
    
    该策略结合机器学习预测和期权交易，实现低风险高收益：
    
    1. 价格预测：使用多因子模型预测期货价格走势
    2. 信号生成：根据预测结果和置信度生成交易信号
    3. 策略选择：基于信号强度选择合适的期权策略
    4. 风险管理：通过希腊字母控制各类风险敞口
    5. 动态调整：根据市场变化动态调整策略参数
    """
    
    def __init__(self, parameters: Optional[PredictionBasedParameters] = None):
        super().__init__(parameters or PredictionBasedParameters())
        self.params = self.parameters
        self.logger = logging.getLogger(__name__)
        
        # 初始化预测模型（这里使用简化版本，实际应该加载训练好的模型）
        self.prediction_model = None
        self.model_last_trained = None
        
        # 策略执行统计
        self.total_predictions = 0
        self.successful_predictions = 0
        self.strategy_performance = {}
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.PREDICTION_BASED
    
    def scan_opportunities(self, options_data: List[OptionData]) -> StrategyResult:
        """
        扫描基于预测的期权交易机会
        """
        start_time = datetime.now()
        opportunities = []
        
        try:
            # 按标的资产分组
            options_by_underlying = {}
            for option in options_data:
                if option.underlying not in options_by_underlying:
                    options_by_underlying[option.underlying] = []
                options_by_underlying[option.underlying].append(option)
            
            # 为每个标的生成预测和策略
            for underlying, underlying_options in options_by_underlying.items():
                try:
                    # 1. 生成价格预测
                    prediction = self._generate_price_prediction(underlying, underlying_options)
                    
                    if prediction.confidence < self.params.min_prediction_confidence:
                        self.logger.debug(f"预测置信度不足: {underlying} - {prediction.confidence}")
                        continue
                    
                    # 2. 筛选合适的期权
                    suitable_options = self._filter_suitable_options(underlying_options, prediction)
                    
                    if len(suitable_options) < 2:  # 至少需要2个期权才能组成策略
                        continue
                    
                    # 3. 生成策略推荐
                    recommendations = self._generate_strategy_recommendations(
                        prediction, suitable_options
                    )
                    
                    # 4. 评估和筛选策略
                    for rec in recommendations:
                        opportunity = self._create_opportunity_from_recommendation(
                            underlying, prediction, rec
                        )
                        
                        if opportunity and self.validate_opportunity(opportunity):
                            opportunities.append(opportunity)
                
                except Exception as e:
                    self.logger.error(f"处理标的 {underlying} 时出错: {e}")
                    continue
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return StrategyResult(
                strategy_name=self.name,
                opportunities=opportunities,
                execution_time=execution_time,
                data_timestamp=datetime.now(),
                success=True,
                metadata={
                    'total_underlyings': len(options_by_underlying),
                    'predictions_generated': len(options_by_underlying),
                    'opportunities_found': len(opportunities),
                    'avg_prediction_confidence': np.mean([
                        op.parameters.get('prediction_confidence', 0) 
                        for op in opportunities
                    ]) if opportunities else 0
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"策略执行失败: {e}", exc_info=True)
            
            return StrategyResult(
                strategy_name=self.name,
                opportunities=[],
                execution_time=execution_time,
                data_timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )
    
    def _generate_price_prediction(
        self, 
        underlying: str, 
        options: List[OptionData]
    ) -> PredictionResult:
        """
        生成价格预测（简化版本）
        
        实际实现中应该：
        1. 获取历史价格数据
        2. 计算技术指标特征
        3. 使用训练好的模型进行预测
        4. 计算置信度和预期波动率
        """
        # 这里使用模拟预测，实际应该调用机器学习模型
        current_price = self._estimate_underlying_price(options)
        
        # 模拟预测逻辑（基于简单技术指标）
        # 实际实现中应该使用复杂的机器学习模型
        np.random.seed(hash(underlying) % 2147483647)  # 确保结果可复现
        
        # 生成预测信号
        signal_prob = np.random.random()
        if signal_prob > 0.7:
            signal = PredictionSignal.BULLISH
            expected_move = 0.03 + np.random.random() * 0.05  # 3-8%上涨
        elif signal_prob < 0.3:
            signal = PredictionSignal.BEARISH
            expected_move = -(0.03 + np.random.random() * 0.05)  # 3-8%下跌
        else:
            signal = PredictionSignal.NEUTRAL
            expected_move = (np.random.random() - 0.5) * 0.02  # ±1%
        
        # 计算预测价格
        predicted_price = current_price * (1 + expected_move)
        
        # 生成置信度（基于信号强度）
        base_confidence = 0.6
        if abs(expected_move) > 0.05:  # 大幅波动，置信度更高
            confidence = base_confidence + 0.2
        else:
            confidence = base_confidence + np.random.random() * 0.2
        
        # 波动率预测（简化）
        volatility_forecast = 0.2 + np.random.random() * 0.3  # 20-50%年化波动率
        
        return PredictionResult(
            signal=signal,
            confidence=min(confidence, 0.95),  # 最高95%置信度
            predicted_price=predicted_price,
            current_price=current_price,
            time_horizon=15,  # 15天预测
            expected_move=expected_move,
            volatility_forecast=volatility_forecast
        )
    
    def _filter_suitable_options(
        self, 
        options: List[OptionData], 
        prediction: PredictionResult
    ) -> List[OptionData]:
        """筛选适合的期权合约"""
        suitable_options = []
        
        for option in options:
            # 检查到期时间
            if not (self.params.min_time_to_expiry <= option.days_to_expiry <= self.params.max_time_to_expiry):
                continue
            
            # 检查行权价范围（相对于当前价格的moneyness）
            moneyness = option.strike_price / prediction.current_price
            if not (self.params.preferred_moneyness_range[0] <= moneyness <= self.params.preferred_moneyness_range[1]):
                continue
            
            # 检查流动性
            if option.volume < self.params.min_liquidity_volume:
                continue
            
            if option.open_interest < self.params.min_open_interest:
                continue
            
            # 检查买卖价差（如果有bid/ask数据）
            if option.bid_price > 0 and option.ask_price > 0:
                spread_pct = (option.ask_price - option.bid_price) / option.mid_price
                if spread_pct > self.params.max_bid_ask_spread:
                    continue
            
            suitable_options.append(option)
        
        return suitable_options
    
    def _generate_strategy_recommendations(
        self, 
        prediction: PredictionResult, 
        options: List[OptionData]
    ) -> List[StrategyRecommendation]:
        """根据预测生成策略推荐"""
        recommendations = []
        
        # 按期权类型分组
        calls = [opt for opt in options if opt.option_type == OptionType.CALL]
        puts = [opt for opt in options if opt.option_type == OptionType.PUT]
        
        # 按行权价排序
        calls.sort(key=lambda x: x.strike_price)
        puts.sort(key=lambda x: x.strike_price)
        
        if prediction.signal == PredictionSignal.BULLISH and self.params.enable_directional_strategies:
            # 看涨策略
            if calls:
                # 买入看涨期权
                recommendations.append(self._create_long_call_strategy(calls, prediction))
                
                # 牛市看涨价差（如果有多个行权价）
                if len(calls) >= 2 and self.params.enable_spread_strategies:
                    recommendations.append(self._create_bull_call_spread(calls, prediction))
        
        elif prediction.signal == PredictionSignal.BEARISH and self.params.enable_directional_strategies:
            # 看跌策略
            if puts:
                # 买入看跌期权
                recommendations.append(self._create_long_put_strategy(puts, prediction))
                
                # 熊市看跌价差
                if len(puts) >= 2 and self.params.enable_spread_strategies:
                    recommendations.append(self._create_bear_put_spread(puts, prediction))
        
        elif prediction.signal == PredictionSignal.NEUTRAL and self.params.enable_volatility_strategies:
            # 中性策略
            if calls and puts:
                # 跨式组合（预期大幅波动）
                if prediction.volatility_forecast > 0.3:
                    recommendations.append(self._create_straddle_strategy(calls, puts, prediction))
                
                # 铁鹰价差（预期小幅波动）
                if prediction.volatility_forecast < 0.25 and len(calls) >= 2 and len(puts) >= 2:
                    recommendations.append(self._create_iron_condor_strategy(calls, puts, prediction))
        
        # 过滤掉None的推荐
        return [rec for rec in recommendations if rec is not None]
    
    def _create_long_call_strategy(
        self, 
        calls: List[OptionData], 
        prediction: PredictionResult
    ) -> Optional[StrategyRecommendation]:
        """创建买入看涨期权策略"""
        if not calls:
            return None
        
        # 选择最接近ATM的看涨期权
        best_call = min(calls, key=lambda x: abs(x.strike_price - prediction.current_price))
        
        actions = [TradingAction(
            instrument=best_call.code,
            action=ActionType.BUY,
            quantity=1,
            price=best_call.market_price
        )]
        
        # 计算盈亏
        max_loss = best_call.market_price
        expected_profit = max(0, prediction.predicted_price - best_call.strike_price - best_call.market_price)
        breakeven = best_call.strike_price + best_call.market_price
        
        # 计算希腊字母
        greeks = {
            'delta': best_call.delta or 0.5,
            'gamma': best_call.gamma or 0.1,
            'theta': best_call.theta or -0.05,
            'vega': best_call.vega or 0.2
        }
        
        return StrategyRecommendation(
            strategy=OptionStrategy.LONG_CALL,
            options=[best_call],
            actions=actions,
            expected_profit=expected_profit,
            max_loss=max_loss,
            breakeven_points=[breakeven],
            greeks=greeks
        )
    
    def _create_long_put_strategy(
        self, 
        puts: List[OptionData], 
        prediction: PredictionResult
    ) -> Optional[StrategyRecommendation]:
        """创建买入看跌期权策略"""
        if not puts:
            return None
        
        # 选择最接近ATM的看跌期权
        best_put = min(puts, key=lambda x: abs(x.strike_price - prediction.current_price))
        
        actions = [TradingAction(
            instrument=best_put.code,
            action=ActionType.BUY,
            quantity=1,
            price=best_put.market_price
        )]
        
        # 计算盈亏
        max_loss = best_put.market_price
        expected_profit = max(0, best_put.strike_price - prediction.predicted_price - best_put.market_price)
        breakeven = best_put.strike_price - best_put.market_price
        
        # 计算希腊字母
        greeks = {
            'delta': best_put.delta or -0.5,
            'gamma': best_put.gamma or 0.1,
            'theta': best_put.theta or -0.05,
            'vega': best_put.vega or 0.2
        }
        
        return StrategyRecommendation(
            strategy=OptionStrategy.LONG_PUT,
            options=[best_put],
            actions=actions,
            expected_profit=expected_profit,
            max_loss=max_loss,
            breakeven_points=[breakeven],
            greeks=greeks
        )
    
    def _create_bull_call_spread(
        self, 
        calls: List[OptionData], 
        prediction: PredictionResult
    ) -> Optional[StrategyRecommendation]:
        """创建牛市看涨价差策略"""
        if len(calls) < 2:
            return None
        
        # 选择两个不同行权价的看涨期权
        # 买入低行权价，卖出高行权价
        low_strike_call = min(calls, key=lambda x: x.strike_price)
        high_strike_calls = [c for c in calls if c.strike_price > low_strike_call.strike_price]
        
        if not high_strike_calls:
            return None
        
        high_strike_call = min(high_strike_calls, key=lambda x: x.strike_price)
        
        actions = [
            TradingAction(
                instrument=low_strike_call.code,
                action=ActionType.BUY,
                quantity=1,
                price=low_strike_call.market_price
            ),
            TradingAction(
                instrument=high_strike_call.code,
                action=ActionType.SELL,
                quantity=1,
                price=high_strike_call.market_price
            )
        ]
        
        # 计算盈亏
        net_premium = low_strike_call.market_price - high_strike_call.market_price
        max_loss = net_premium
        max_gain = (high_strike_call.strike_price - low_strike_call.strike_price) - net_premium
        expected_profit = max(0, min(
            prediction.predicted_price - low_strike_call.strike_price,
            high_strike_call.strike_price - low_strike_call.strike_price
        ) - net_premium)
        
        breakeven = low_strike_call.strike_price + net_premium
        
        # 计算希腊字母
        greeks = {
            'delta': (low_strike_call.delta or 0.5) - (high_strike_call.delta or 0.3),
            'gamma': (low_strike_call.gamma or 0.1) - (high_strike_call.gamma or 0.05),
            'theta': (low_strike_call.theta or -0.05) - (high_strike_call.theta or -0.03),
            'vega': (low_strike_call.vega or 0.2) - (high_strike_call.vega or 0.15)
        }
        
        return StrategyRecommendation(
            strategy=OptionStrategy.BULL_CALL_SPREAD,
            options=[low_strike_call, high_strike_call],
            actions=actions,
            expected_profit=expected_profit,
            max_loss=max_loss,
            breakeven_points=[breakeven],
            greeks=greeks
        )
    
    def _create_bear_put_spread(
        self, 
        puts: List[OptionData], 
        prediction: PredictionResult
    ) -> Optional[StrategyRecommendation]:
        """创建熊市看跌价差策略"""
        if len(puts) < 2:
            return None
        
        # 买入高行权价，卖出低行权价
        high_strike_put = max(puts, key=lambda x: x.strike_price)
        low_strike_puts = [p for p in puts if p.strike_price < high_strike_put.strike_price]
        
        if not low_strike_puts:
            return None
        
        low_strike_put = max(low_strike_puts, key=lambda x: x.strike_price)
        
        actions = [
            TradingAction(
                instrument=high_strike_put.code,
                action=ActionType.BUY,
                quantity=1,
                price=high_strike_put.market_price
            ),
            TradingAction(
                instrument=low_strike_put.code,
                action=ActionType.SELL,
                quantity=1,
                price=low_strike_put.market_price
            )
        ]
        
        # 计算盈亏
        net_premium = high_strike_put.market_price - low_strike_put.market_price
        max_loss = net_premium
        max_gain = (high_strike_put.strike_price - low_strike_put.strike_price) - net_premium
        expected_profit = max(0, min(
            high_strike_put.strike_price - prediction.predicted_price,
            high_strike_put.strike_price - low_strike_put.strike_price
        ) - net_premium)
        
        breakeven = high_strike_put.strike_price - net_premium
        
        # 计算希腊字母
        greeks = {
            'delta': (high_strike_put.delta or -0.5) - (low_strike_put.delta or -0.3),
            'gamma': (high_strike_put.gamma or 0.1) - (low_strike_put.gamma or 0.05),
            'theta': (high_strike_put.theta or -0.05) - (low_strike_put.theta or -0.03),
            'vega': (high_strike_put.vega or 0.2) - (low_strike_put.vega or 0.15)
        }
        
        return StrategyRecommendation(
            strategy=OptionStrategy.BEAR_PUT_SPREAD,
            options=[high_strike_put, low_strike_put],
            actions=actions,
            expected_profit=expected_profit,
            max_loss=max_loss,
            breakeven_points=[breakeven],
            greeks=greeks
        )
    
    def _create_straddle_strategy(
        self, 
        calls: List[OptionData], 
        puts: List[OptionData], 
        prediction: PredictionResult
    ) -> Optional[StrategyRecommendation]:
        """创建跨式组合策略"""
        if not calls or not puts:
            return None
        
        # 寻找相同行权价的看涨和看跌期权
        atm_strike = prediction.current_price
        
        # 找到最接近ATM的期权
        best_call = min(calls, key=lambda x: abs(x.strike_price - atm_strike))
        best_put = min(puts, key=lambda x: abs(x.strike_price - best_call.strike_price))
        
        # 确保是相同行权价（或非常接近）
        if abs(best_call.strike_price - best_put.strike_price) > atm_strike * 0.02:  # 2%误差
            return None
        
        actions = [
            TradingAction(
                instrument=best_call.code,
                action=ActionType.BUY,
                quantity=1,
                price=best_call.market_price
            ),
            TradingAction(
                instrument=best_put.code,
                action=ActionType.BUY,
                quantity=1,
                price=best_put.market_price
            )
        ]
        
        # 计算盈亏
        total_premium = best_call.market_price + best_put.market_price
        max_loss = total_premium
        
        # 跨式组合在大幅波动时盈利
        strike_price = best_call.strike_price
        expected_move_abs = abs(prediction.predicted_price - prediction.current_price)
        expected_profit = max(0, expected_move_abs - total_premium)
        
        # 两个盈亏平衡点
        breakeven_upper = strike_price + total_premium
        breakeven_lower = strike_price - total_premium
        
        # 计算希腊字母
        greeks = {
            'delta': (best_call.delta or 0.5) + (best_put.delta or -0.5),  # 接近0
            'gamma': (best_call.gamma or 0.1) + (best_put.gamma or 0.1),
            'theta': (best_call.theta or -0.05) + (best_put.theta or -0.05),
            'vega': (best_call.vega or 0.2) + (best_put.vega or 0.2)
        }
        
        return StrategyRecommendation(
            strategy=OptionStrategy.STRADDLE,
            options=[best_call, best_put],
            actions=actions,
            expected_profit=expected_profit,
            max_loss=max_loss,
            breakeven_points=[breakeven_lower, breakeven_upper],
            greeks=greeks
        )
    
    def _create_iron_condor_strategy(
        self, 
        calls: List[OptionData], 
        puts: List[OptionData], 
        prediction: PredictionResult
    ) -> Optional[StrategyRecommendation]:
        """创建铁鹰价差策略（中性策略，预期小幅波动）"""
        if len(calls) < 2 or len(puts) < 2:
            return None
        
        # 铁鹰策略：卖出ATM跨式，买入保护性OTM期权
        current_price = prediction.current_price
        
        # 选择期权
        calls_sorted = sorted(calls, key=lambda x: x.strike_price)
        puts_sorted = sorted(puts, key=lambda x: x.strike_price, reverse=True)
        
        # ATM附近的期权（卖出）
        atm_call = min(calls, key=lambda x: abs(x.strike_price - current_price))
        atm_put = min(puts, key=lambda x: abs(x.strike_price - current_price))
        
        # OTM保护性期权（买入）
        otm_calls = [c for c in calls if c.strike_price > atm_call.strike_price]
        otm_puts = [p for p in puts if p.strike_price < atm_put.strike_price]
        
        if not otm_calls or not otm_puts:
            return None
        
        otm_call = min(otm_calls, key=lambda x: x.strike_price)  # 最近的OTM call
        otm_put = max(otm_puts, key=lambda x: x.strike_price)   # 最近的OTM put
        
        actions = [
            # 卖出ATM跨式
            TradingAction(
                instrument=atm_call.code,
                action=ActionType.SELL,
                quantity=1,
                price=atm_call.market_price
            ),
            TradingAction(
                instrument=atm_put.code,
                action=ActionType.SELL,
                quantity=1,
                price=atm_put.market_price
            ),
            # 买入OTM保护
            TradingAction(
                instrument=otm_call.code,
                action=ActionType.BUY,
                quantity=1,
                price=otm_call.market_price
            ),
            TradingAction(
                instrument=otm_put.code,
                action=ActionType.BUY,
                quantity=1,
                price=otm_put.market_price
            )
        ]
        
        # 计算盈亏
        net_credit = (atm_call.market_price + atm_put.market_price) - \
                    (otm_call.market_price + otm_put.market_price)
        
        max_gain = net_credit  # 价格在ATM附近时的最大收益
        
        # 最大损失是价差减去净收益
        call_spread_width = otm_call.strike_price - atm_call.strike_price
        put_spread_width = atm_put.strike_price - otm_put.strike_price
        max_loss = max(call_spread_width, put_spread_width) - net_credit
        
        # 如果预测价格在ATM附近，期望收益为净收益
        if abs(prediction.predicted_price - current_price) / current_price < 0.02:
            expected_profit = net_credit
        else:
            expected_profit = 0  # 价格偏离太多时无收益
        
        # 盈亏平衡点
        breakeven_upper = atm_call.strike_price + net_credit
        breakeven_lower = atm_put.strike_price - net_credit
        
        # 计算希腊字母
        greeks = {
            'delta': -(atm_call.delta or 0.5) - (atm_put.delta or -0.5) + \
                    (otm_call.delta or 0.2) + (otm_put.delta or -0.2),
            'gamma': -(atm_call.gamma or 0.1) - (atm_put.gamma or 0.1) + \
                    (otm_call.gamma or 0.05) + (otm_put.gamma or 0.05),
            'theta': -(atm_call.theta or -0.05) - (atm_put.theta or -0.05) + \
                    (otm_call.theta or -0.02) + (otm_put.theta or -0.02),
            'vega': -(atm_call.vega or 0.2) - (atm_put.vega or 0.2) + \
                   (otm_call.vega or 0.1) + (otm_put.vega or 0.1)
        }
        
        return StrategyRecommendation(
            strategy=OptionStrategy.IRON_CONDOR,
            options=[atm_call, atm_put, otm_call, otm_put],
            actions=actions,
            expected_profit=expected_profit,
            max_loss=max_loss,
            breakeven_points=[breakeven_lower, breakeven_upper],
            greeks=greeks
        )
    
    def _create_opportunity_from_recommendation(
        self, 
        underlying: str, 
        prediction: PredictionResult, 
        recommendation: StrategyRecommendation
    ) -> Optional[ArbitrageOpportunity]:
        """从策略推荐创建套利机会"""
        try:
            # 风险检查
            if not self._validate_greeks(recommendation.greeks):
                return None
            
            if recommendation.max_loss > self.params.max_single_position_risk * 100000:  # 假设总资金100万
                return None
            
            # 创建交易动作
            actions = []
            for action in recommendation.actions:
                actions.append({
                    'instrument': action.instrument,
                    'action': action.action.value,
                    'quantity': action.quantity,
                    'price': action.price,
                    'reasoning': f"{recommendation.strategy.value}策略组成部分"
                })
            
            # 获取市场价格和成交量
            market_prices = {}
            volumes = {}
            for option in recommendation.options:
                market_prices[option.code] = option.market_price
                volumes[option.code] = option.volume
            
            opportunity = ArbitrageOpportunity(
                id=f"pred_{recommendation.strategy.value}_{underlying}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                strategy_type=self.strategy_type,
                timestamp=datetime.now(),
                instruments=[opt.code for opt in recommendation.options],
                underlying=underlying,
                expected_profit=recommendation.expected_profit,
                profit_margin=recommendation.expected_profit / max(recommendation.max_loss, 1),
                confidence_score=prediction.confidence,
                max_loss=recommendation.max_loss,
                risk_score=self._calculate_risk_score_from_greeks(recommendation.greeks),
                days_to_expiry=min(opt.days_to_expiry for opt in recommendation.options),
                market_prices=market_prices,
                volumes=volumes,
                actions=actions,
                data_source="prediction_based_strategy",
                parameters={
                    'prediction_signal': prediction.signal.value,
                    'prediction_confidence': prediction.confidence,
                    'predicted_price': prediction.predicted_price,
                    'current_price': prediction.current_price,
                    'expected_move': prediction.expected_move,
                    'volatility_forecast': prediction.volatility_forecast,
                    'option_strategy': recommendation.strategy.value,
                    'breakeven_points': recommendation.breakeven_points,
                    'greeks': recommendation.greeks,
                    'time_horizon': prediction.time_horizon
                }
            )
            
            return opportunity
            
        except Exception as e:
            self.logger.error(f"创建机会时出错: {e}")
            return None
    
    def _validate_greeks(self, greeks: Dict[str, float]) -> bool:
        """验证希腊字母是否在可接受范围内"""
        delta = abs(greeks.get('delta', 0))
        if delta > self.params.max_delta_exposure:
            return False
        
        vega = abs(greeks.get('vega', 0))
        if vega > self.params.max_vega_exposure:
            return False
        
        theta = greeks.get('theta', 0)
        if theta < self.params.max_theta_exposure:  # theta通常为负值
            return False
        
        return True
    
    def _calculate_risk_score_from_greeks(self, greeks: Dict[str, float]) -> float:
        """基于希腊字母计算风险评分"""
        delta_risk = abs(greeks.get('delta', 0)) / self.params.max_delta_exposure
        vega_risk = abs(greeks.get('vega', 0)) / self.params.max_vega_exposure
        theta_risk = abs(greeks.get('theta', 0)) / abs(self.params.max_theta_exposure)
        
        # 综合风险评分
        total_risk = (delta_risk * 0.4 + vega_risk * 0.3 + theta_risk * 0.3)
        return min(total_risk, 1.0)
    
    def _estimate_underlying_price(self, options: List[OptionData]) -> float:
        """估算标的资产价格"""
        if not options:
            return 1000.0  # 默认价格
        
        # 使用平均行权价作为近似价格
        strikes = [opt.strike_price for opt in options]
        return np.mean(strikes)
    
    def calculate_profit(self, options: List[OptionData], actions: List[TradingAction]) -> float:
        """计算期望收益"""
        # 这里应该根据预测价格计算各种情况下的收益
        # 简化实现
        total_premium = 0
        for action in actions:
            option = next((opt for opt in options if opt.code == action.instrument), None)
            if option:
                if action.action == ActionType.BUY:
                    total_premium -= action.price or option.market_price
                else:
                    total_premium += action.price or option.market_price
        
        # 这里应该有更复杂的收益计算逻辑
        return abs(total_premium) * 0.2  # 简化：假设20%的收益率
    
    def assess_risk(self, options: List[OptionData], actions: List[TradingAction]) -> RiskMetrics:
        """评估策略风险"""
        max_loss = 0
        max_gain = 0
        
        # 计算最大可能损失
        for action in actions:
            option = next((opt for opt in options if opt.code == action.instrument), None)
            if option:
                position_value = (action.price or option.market_price) * action.quantity
                if action.action == ActionType.BUY:
                    max_loss += position_value  # 买入期权的最大损失是权利金
                else:
                    max_gain += position_value  # 卖出期权的最大收益是权利金
        
        # 基于策略类型调整风险评估
        probability_profit = 0.6  # 基础概率
        
        # 风险等级评估
        if max_loss < 1000:
            risk_level = RiskLevel.LOW
        elif max_loss < 5000:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.HIGH
        
        expected_return = probability_profit * max_gain - (1 - probability_profit) * max_loss
        
        return RiskMetrics(
            max_loss=max_loss,
            max_gain=max_gain,
            probability_profit=probability_profit,
            expected_return=expected_return,
            risk_level=risk_level,
            liquidity_risk=0.1,  # 基于期权流动性
            time_decay_risk=0.2,  # 时间衰减风险
            volatility_risk=0.15  # 波动率风险
        )
    
    def get_prediction_performance(self) -> Dict[str, Any]:
        """获取预测性能统计"""
        if self.total_predictions == 0:
            return {'accuracy': 0, 'total_predictions': 0}
        
        return {
            'accuracy': self.successful_predictions / self.total_predictions,
            'total_predictions': self.total_predictions,
            'successful_predictions': self.successful_predictions,
            'strategy_performance': self.strategy_performance
        }
    
    def update_prediction_result(self, prediction_id: str, actual_result: bool):
        """更新预测结果（用于模型改进）"""
        self.total_predictions += 1
        if actual_result:
            self.successful_predictions += 1
        
        # 这里可以添加更详细的结果跟踪逻辑
        # 用于后续的模型优化和策略改进