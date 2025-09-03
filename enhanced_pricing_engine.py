#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版期权定价引擎
基于legacy_logic分析结果的优化实现
包含数值稳定性改进、性能优化和风险控制增强
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import erf, erfc
# import numba
# from numba import jit
# 注释掉numba依赖以便演示运行

def jit(**kwargs):
    """Mock decorator for demonstration"""
    def decorator(func):
        return func
    return decorator
import math
from typing import Union, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class OptionPricingResult:
    """期权定价结果数据类"""
    theoretical_price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    implied_volatility: Optional[float] = None
    pricing_error: Optional[float] = None


class EnhancedBlackScholesEngine:
    """增强版Black-Scholes定价引擎"""
    
    def __init__(self, use_high_precision: bool = True):
        self.use_high_precision = use_high_precision
        self.precision_threshold = 1e-10
        
    @staticmethod
    @jit(nopython=True)
    def _stable_normal_cdf(x: float) -> float:
        """数值稳定的正态累积分布函数"""
        if x > 6.0:
            return 1.0
        elif x < -6.0:
            return 0.0
        else:
            return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))
    
    @staticmethod
    @jit(nopython=True)
    def _stable_normal_pdf(x: float) -> float:
        """数值稳定的正态概率密度函数"""
        if abs(x) > 6.0:
            return 0.0
        else:
            return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    
    def black_scholes_call(self, S: float, K: float, T: float, 
                          r: float, sigma: float) -> float:
        """
        增强版看涨期权定价
        
        Args:
            S: 标的价格
            K: 行权价
            T: 到期时间（年）
            r: 无风险利率
            sigma: 波动率
            
        Returns:
            期权理论价格
        """
        try:
            # 边界条件检查
            if T <= 0:
                return max(S - K, 0)
            if sigma <= 0:
                return max(S - K * math.exp(-r * T), 0)
            if S <= 0 or K <= 0:
                return 0.0
            
            # 极端情况处理
            if S / K > 1000 or S / K < 0.001:
                return self._asymptotic_call_price(S, K, T, r, sigma)
            
            return self._standard_bs_call(S, K, T, r, sigma)
            
        except (OverflowError, ZeroDivisionError, ValueError):
            return max(S - K * math.exp(-r * T), 0)
    
    @jit(forceobj=True)
    def _standard_bs_call(self, S: float, K: float, T: float, 
                         r: float, sigma: float) -> float:
        """标准Black-Scholes计算"""
        sqrt_T = math.sqrt(T)
        sigma_sqrt_T = sigma * sqrt_T
        
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T
        
        # 使用稳定的CDF计算
        N_d1 = self._stable_normal_cdf(d1)
        N_d2 = self._stable_normal_cdf(d2)
        
        price = S * N_d1 - K * math.exp(-r * T) * N_d2
        return max(price, 0.0)
    
    def _asymptotic_call_price(self, S: float, K: float, T: float, 
                              r: float, sigma: float) -> float:
        """极端情况下的渐近价格"""
        if S / K > 1000:  # 深度实值
            return S - K * math.exp(-r * T)
        else:  # 深度虚值
            return 0.0
    
    def black_scholes_put(self, S: float, K: float, T: float, 
                         r: float, sigma: float) -> float:
        """
        增强版看跌期权定价
        使用看涨-看跌平价关系提高数值稳定性
        """
        call_price = self.black_scholes_call(S, K, T, r, sigma)
        put_price = call_price - S + K * math.exp(-r * T)
        return max(put_price, 0.0)
    
    def calculate_greeks(self, S: float, K: float, T: float, 
                        r: float, sigma: float, option_type: str = 'call') -> dict:
        """
        计算期权希腊字母
        
        Returns:
            包含所有希腊字母的字典
        """
        try:
            if T <= 0:
                return self._expiry_greeks(S, K, option_type)
            
            sqrt_T = math.sqrt(T)
            d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
            d2 = d1 - sigma * sqrt_T
            
            # 使用稳定的概率函数
            N_d1 = self._stable_normal_cdf(d1)
            N_d2 = self._stable_normal_cdf(d2)
            n_d1 = self._stable_normal_pdf(d1)
            
            if option_type.lower() == 'call':
                delta = N_d1
                theta = (-S * n_d1 * sigma / (2 * sqrt_T) 
                        - r * K * math.exp(-r * T) * N_d2) / 365.25
            else:  # put
                delta = N_d1 - 1
                theta = (-S * n_d1 * sigma / (2 * sqrt_T) 
                        + r * K * math.exp(-r * T) * (1 - N_d2)) / 365.25
            
            # 共同的希腊字母
            gamma = n_d1 / (S * sigma * sqrt_T)
            vega = S * n_d1 * sqrt_T / 100  # 除以100得到1%波动率变化的影响
            rho = (K * T * math.exp(-r * T) * 
                   (N_d2 if option_type.lower() == 'call' else (N_d2 - 1))) / 100
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }
            
        except (OverflowError, ZeroDivisionError, ValueError):
            return self._default_greeks()
    
    def _expiry_greeks(self, S: float, K: float, option_type: str) -> dict:
        """到期时的希腊字母"""
        if option_type.lower() == 'call':
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        
        return {
            'delta': delta,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
    
    def _default_greeks(self) -> dict:
        """默认希腊字母值"""
        return {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }


class RobustImpliedVolatility:
    """鲁棒隐含波动率计算器"""
    
    def __init__(self, max_iterations: int = 50, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.bs_engine = EnhancedBlackScholesEngine()
    
    def calculate(self, market_price: float, S: float, K: float, T: float,
                  r: float, option_type: str = 'call') -> Optional[float]:
        """
        使用多种方法计算隐含波动率
        
        优先级：
        1. 改进牛顿法（多起点）
        2. Brent方法
        3. 二分法
        """
        # 基本检查
        if market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
            return None
        
        # 内在价值检查
        if option_type.lower() == 'call':
            intrinsic = max(S - K * math.exp(-r * T), 0)
        else:
            intrinsic = max(K * math.exp(-r * T) - S, 0)
        
        if market_price <= intrinsic:
            return 0.001  # 极小的隐含波动率
        
        # 尝试多起点牛顿法
        initial_guesses = [0.2, 0.5, 1.0, 0.1, 1.5]
        for initial_vol in initial_guesses:
            result = self._newton_raphson(market_price, S, K, T, r, 
                                         option_type, initial_vol)
            if result is not None:
                return result
        
        # 备用方法：二分法
        return self._bisection_method(market_price, S, K, T, r, option_type)
    
    def _newton_raphson(self, target_price: float, S: float, K: float, T: float,
                       r: float, option_type: str, initial_vol: float) -> Optional[float]:
        """改进的牛顿-拉夫逊方法"""
        vol = initial_vol
        
        for i in range(self.max_iterations):
            try:
                # 计算理论价格和Vega
                if option_type.lower() == 'call':
                    theo_price = self.bs_engine.black_scholes_call(S, K, T, r, vol)
                else:
                    theo_price = self.bs_engine.black_scholes_put(S, K, T, r, vol)
                
                price_diff = theo_price - target_price
                
                if abs(price_diff) < self.tolerance:
                    return vol
                
                # 计算Vega
                greeks = self.bs_engine.calculate_greeks(S, K, T, r, vol, option_type)
                vega = greeks['vega'] * 100  # 转换回小数形式
                
                if abs(vega) < 1e-10:  # Vega太小，退出
                    break
                
                # 牛顿法更新
                vol_new = vol - price_diff / vega
                
                # 约束检查
                vol_new = max(0.001, min(vol_new, 5.0))
                
                # 收敛检查
                if abs(vol_new - vol) < self.tolerance:
                    return vol_new
                
                vol = vol_new
                
            except (OverflowError, ZeroDivisionError, ValueError):
                break
        
        return None
    
    def _bisection_method(self, target_price: float, S: float, K: float, T: float,
                         r: float, option_type: str) -> Optional[float]:
        """二分法求解隐含波动率"""
        vol_low, vol_high = 0.001, 5.0
        
        for _ in range(self.max_iterations):
            vol_mid = (vol_low + vol_high) / 2
            
            if option_type.lower() == 'call':
                price_mid = self.bs_engine.black_scholes_call(S, K, T, r, vol_mid)
            else:
                price_mid = self.bs_engine.black_scholes_put(S, K, T, r, vol_mid)
            
            if abs(price_mid - target_price) < self.tolerance:
                return vol_mid
            
            if price_mid < target_price:
                vol_low = vol_mid
            else:
                vol_high = vol_mid
                
            if vol_high - vol_low < self.tolerance:
                return vol_mid
        
        return (vol_low + vol_high) / 2


class VectorizedOptionPricer:
    """向量化期权定价器"""
    
    def __init__(self):
        self.bs_engine = EnhancedBlackScholesEngine()
        self.iv_calculator = RobustImpliedVolatility()
    
    @staticmethod
    @jit(nopython=True)
    def _vectorized_bs_core(S_array: np.ndarray, K_array: np.ndarray, 
                           T_array: np.ndarray, r: float, 
                           sigma_array: np.ndarray) -> np.ndarray:
        """向量化的Black-Scholes核心计算（JIT编译）"""
        n = len(S_array)
        prices = np.zeros(n)
        
        for i in range(n):
            S, K, T, sigma = S_array[i], K_array[i], T_array[i], sigma_array[i]
            
            if T <= 0:
                prices[i] = max(S - K, 0)
                continue
            if sigma <= 0 or S <= 0 or K <= 0:
                continue
                
            try:
                sqrt_T = math.sqrt(T)
                d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
                d2 = d1 - sigma * sqrt_T
                
                # 简化的CDF计算（JIT友好）
                if d1 > 6:
                    N_d1 = 1.0
                elif d1 < -6:
                    N_d1 = 0.0
                else:
                    N_d1 = 0.5 * (1.0 + erf(d1 / math.sqrt(2.0)))
                
                if d2 > 6:
                    N_d2 = 1.0
                elif d2 < -6:
                    N_d2 = 0.0
                else:
                    N_d2 = 0.5 * (1.0 + erf(d2 / math.sqrt(2.0)))
                
                prices[i] = max(S * N_d1 - K * math.exp(-r * T) * N_d2, 0)
                
            except:
                prices[i] = max(S - K * math.exp(-r * T), 0)
        
        return prices
    
    def batch_pricing(self, options_df: pd.DataFrame, r: float = 0.03) -> pd.DataFrame:
        """
        批量期权定价
        
        Args:
            options_df: 包含期权信息的DataFrame
            r: 无风险利率
            
        Returns:
            包含理论价格的DataFrame
        """
        # 数据准备
        S_array = options_df['underlying_price'].values
        K_array = options_df['exercise_price'].values
        T_array = (options_df['days_to_expiry'] / 365.25).values
        
        # 估算初始波动率
        sigma_array = np.full(len(options_df), 0.3)
        
        # 批量计算理论价格
        theoretical_prices = self._vectorized_bs_core(S_array, K_array, T_array, 
                                                     r, sigma_array)
        
        # 计算隐含波动率（如果有市场价格）
        implied_vols = []
        if 'market_price' in options_df.columns:
            for i, row in options_df.iterrows():
                iv = self.iv_calculator.calculate(
                    row['market_price'], row['underlying_price'], 
                    row['exercise_price'], row['days_to_expiry'] / 365.25,
                    r, row.get('call_put', 'C').lower()
                )
                implied_vols.append(iv)
        
        # 构建结果DataFrame
        result_df = options_df.copy()
        result_df['theoretical_price'] = theoretical_prices
        if implied_vols:
            result_df['implied_volatility'] = implied_vols
        
        return result_df


class ArbitrageDetector:
    """套利机会检测器"""
    
    def __init__(self, pricing_threshold: float = 0.05, 
                 volatility_threshold: float = 0.1):
        self.pricing_threshold = pricing_threshold
        self.volatility_threshold = volatility_threshold
        self.pricer = VectorizedOptionPricer()
    
    def find_pricing_arbitrage(self, options_df: pd.DataFrame) -> pd.DataFrame:
        """
        发现定价套利机会（增强版）
        包含动态阈值和风险调整
        """
        if 'market_price' not in options_df.columns:
            return pd.DataFrame()
        
        # 批量定价
        priced_df = self.pricer.batch_pricing(options_df)
        
        # 计算定价偏差
        priced_df['price_deviation'] = (
            (priced_df['market_price'] - priced_df['theoretical_price']) / 
            priced_df['theoretical_price']
        )
        
        # 动态阈值计算
        market_vol = priced_df['price_deviation'].std()
        dynamic_threshold = max(self.pricing_threshold, market_vol * 2)
        
        # 筛选套利机会
        arbitrage_ops = priced_df[
            (abs(priced_df['price_deviation']) > dynamic_threshold) &
            (priced_df['market_price'] > 0) &
            (priced_df['theoretical_price'] > 0) &
            (priced_df.get('volume', 0) > 10)  # 最小成交量要求
        ].copy()
        
        if not arbitrage_ops.empty:
            # 添加风险评分
            arbitrage_ops['risk_score'] = self._calculate_risk_score(arbitrage_ops)
            arbitrage_ops = arbitrage_ops.sort_values('risk_score')
        
        return arbitrage_ops
    
    def _calculate_risk_score(self, df: pd.DataFrame) -> np.ndarray:
        """计算套利机会的风险评分"""
        # 基础因子
        price_risk = abs(df['price_deviation']) * 10  # 价格偏差风险
        liquidity_risk = 1 / np.log1p(df.get('volume', 1))  # 流动性风险
        time_risk = 1 / np.sqrt(df.get('days_to_expiry', 30))  # 时间风险
        
        # 综合风险评分（越低越好）
        risk_score = price_risk + liquidity_risk + time_risk
        
        return risk_score


def demo_enhanced_pricing():
    """演示增强版定价引擎的功能"""
    print("增强版期权定价引擎演示")
    print("=" * 50)
    
    # 创建测试数据
    test_options = pd.DataFrame({
        'ts_code': ['TEST001C', 'TEST001P', 'TEST002C'],
        'underlying_price': [100.0, 100.0, 100.0],
        'exercise_price': [100.0, 100.0, 110.0],
        'days_to_expiry': [30, 30, 30],
        'call_put': ['C', 'P', 'C'],
        'market_price': [5.2, 4.8, 1.5],
        'volume': [100, 150, 50]
    })
    
    # 初始化定价器
    pricer = VectorizedOptionPricer()
    
    # 批量定价
    print("批量定价结果:")
    priced_options = pricer.batch_pricing(test_options)
    
    for _, row in priced_options.iterrows():
        print(f"{row['ts_code']}: 理论价格 {row['theoretical_price']:.3f}, "
              f"市场价格 {row['market_price']:.3f}")
        if 'implied_volatility' in row:
            print(f"  隐含波动率: {row['implied_volatility']*100:.2f}%")
    
    # 套利检测
    print("\n套利机会检测:")
    detector = ArbitrageDetector()
    arbitrage_ops = detector.find_pricing_arbitrage(priced_options)
    
    if not arbitrage_ops.empty:
        print(f"发现 {len(arbitrage_ops)} 个套利机会:")
        for _, op in arbitrage_ops.iterrows():
            action = "买入" if op['price_deviation'] < 0 else "卖出"
            print(f"  {action} {op['ts_code']}: 偏差 {op['price_deviation']*100:.2f}%, "
                  f"风险评分 {op['risk_score']:.3f}")
    else:
        print("未发现明显套利机会")


if __name__ == "__main__":
    demo_enhanced_pricing()