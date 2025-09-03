#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期权套利机会发现工具
基于tushare数据识别各类期权套利机会
"""

import os
import sys
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import math
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


def load_env_file():
    """加载 .env 文件中的环境变量"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


def initialize_tushare():
    """初始化 tushare"""
    load_env_file()
    token = os.getenv('TUSHARE_TOKEN')
    if not token:
        print("错误: 未找到环境变量 TUSHARE_TOKEN")
        sys.exit(1)
    
    ts.set_token(token)
    return ts.pro_api()


def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes看涨期权定价"""
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return max(call_price, 0)
    except:
        return 0


def black_scholes_put(S, K, T, r, sigma):
    """Black-Scholes看跌期权定价"""
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return max(put_price, 0)
    except:
        return 0


def implied_volatility(option_price, S, K, T, r, option_type='call', max_iter=100):
    """计算隐含波动率（牛顿法）"""
    try:
        if T <= 0 or S <= 0 or K <= 0 or option_price <= 0:
            return 0
        
        # 初始猜测
        sigma = 0.3
        
        for i in range(max_iter):
            if option_type == 'call':
                price = black_scholes_call(S, K, T, r, sigma)
                # Vega计算
                d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                vega = S * norm.pdf(d1) * np.sqrt(T)
            else:
                price = black_scholes_put(S, K, T, r, sigma)
                d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                vega = S * norm.pdf(d1) * np.sqrt(T)
            
            if abs(price - option_price) < 0.0001 or vega == 0:
                break
                
            # 牛顿法更新
            sigma = sigma - (price - option_price) / vega
            sigma = max(sigma, 0.001)  # 避免负波动率
            
        return max(sigma, 0)
    except:
        return 0


def get_option_data_with_pricing(pro, max_days=60):
    """获取期权数据并计算理论价格"""
    print(f"获取期权数据并计算理论价格...")
    
    try:
        # 获取期权基础数据
        options = pro.opt_basic()
        if options.empty:
            return pd.DataFrame()
        
        # 筛选近期到期
        options['delist_date_dt'] = pd.to_datetime(options['delist_date'])
        today = datetime.now()
        options['days_to_expiry'] = (options['delist_date_dt'] - today).dt.days
        
        near_options = options[
            (options['days_to_expiry'] > 0) & 
            (options['days_to_expiry'] <= max_days)
        ].copy()
        
        if near_options.empty:
            print("未找到近期到期的期权")
            return pd.DataFrame()
        
        # 获取行情数据
        print(f"获取期权行情数据...")
        market_data = get_latest_market_data(pro)
        
        if market_data.empty:
            print("未能获取行情数据")
            return pd.DataFrame()
        
        # 合并数据
        full_data = near_options.merge(
            market_data[['ts_code', 'close', 'vol', 'oi', 'trade_date']], 
            on='ts_code', 
            how='left'
        )
        
        # 过滤有行情的期权
        full_data = full_data[full_data['close'].notna() & (full_data['close'] > 0)].copy()
        
        if full_data.empty:
            print("没有有效的期权行情数据")
            return pd.DataFrame()
        
        print(f"找到 {len(full_data)} 个有行情的近期期权")
        
        # 获取标的价格（这里简化处理，实际需要获取对应的期货价格）
        full_data = estimate_underlying_prices(full_data)
        
        # 计算理论价格
        full_data = calculate_theoretical_prices(full_data)
        
        return full_data
        
    except Exception as e:
        print(f"获取期权数据失败: {e}")
        return pd.DataFrame()


def get_latest_market_data(pro):
    """获取最新的期权行情数据"""
    try:
        # 尝试获取最近几天的数据
        for days_back in range(1, 5):
            trade_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
            try:
                daily_data = pro.opt_daily(trade_date=trade_date)
                if not daily_data.empty:
                    print(f"获取到 {trade_date} 的行情数据")
                    return daily_data
            except:
                continue
        return pd.DataFrame()
    except:
        return pd.DataFrame()


def estimate_underlying_prices(options_df):
    """估算标的资产价格（简化方法）"""
    options_df = options_df.copy()
    
    # 从期权代码中提取标的信息
    options_df['underlying'] = options_df['ts_code'].str.extract(r'([A-Z]+\d{4})', expand=False)
    
    # 对于每个标的，使用平价期权估算标的价格
    estimated_prices = {}
    
    for underlying in options_df['underlying'].unique():
        if pd.isna(underlying):
            continue
            
        underlying_options = options_df[options_df['underlying'] == underlying]
        
        # 寻找相同行权价的看涨和看跌期权
        for strike in underlying_options['exercise_price'].unique():
            if pd.isna(strike):
                continue
                
            strike_options = underlying_options[underlying_options['exercise_price'] == strike]
            
            calls = strike_options[strike_options['call_put'] == 'C']
            puts = strike_options[strike_options['call_put'] == 'P']
            
            if not calls.empty and not puts.empty:
                call_price = calls['close'].iloc[0]
                put_price = puts['close'].iloc[0]
                
                # 使用期权平价关系估算标的价格: S = C - P + K * e^(-r*T)
                # 简化: S ≈ C - P + K (假设无risk-free rate)
                estimated_S = call_price - put_price + strike
                if estimated_S > 0:
                    estimated_prices[underlying] = estimated_S
                    break
    
    # 如果无法通过平价关系估算，使用行权价作为粗略估计
    for underlying in options_df['underlying'].unique():
        if underlying not in estimated_prices and not pd.isna(underlying):
            underlying_options = options_df[options_df['underlying'] == underlying]
            atm_options = underlying_options.loc[underlying_options['exercise_price'].idxmin()]
            estimated_prices[underlying] = atm_options['exercise_price']
    
    # 将估算的价格分配给期权
    options_df['underlying_price'] = options_df['underlying'].map(estimated_prices)
    
    return options_df


def calculate_theoretical_prices(options_df):
    """计算理论价格和隐含波动率"""
    options_df = options_df.copy()
    
    # 参数设置
    risk_free_rate = 0.03  # 假设3%无风险利率
    
    theoretical_prices = []
    implied_vols = []
    price_deviations = []
    
    for _, row in options_df.iterrows():
        try:
            S = row['underlying_price']
            K = row['exercise_price'] 
            T = row['days_to_expiry'] / 365.0
            market_price = row['close']
            option_type = 'call' if row['call_put'] == 'C' else 'put'
            
            if pd.isna(S) or pd.isna(K) or pd.isna(market_price) or S <= 0 or K <= 0 or T <= 0:
                theoretical_prices.append(np.nan)
                implied_vols.append(np.nan)
                price_deviations.append(np.nan)
                continue
            
            # 计算隐含波动率
            iv = implied_volatility(market_price, S, K, T, risk_free_rate, option_type)
            implied_vols.append(iv)
            
            # 使用隐含波动率计算理论价格
            if option_type == 'call':
                theo_price = black_scholes_call(S, K, T, risk_free_rate, iv)
            else:
                theo_price = black_scholes_put(S, K, T, risk_free_rate, iv)
            
            theoretical_prices.append(theo_price)
            
            # 计算价格偏差
            if theo_price > 0:
                deviation = (market_price - theo_price) / theo_price
            else:
                deviation = 0
            price_deviations.append(deviation)
            
        except Exception as e:
            theoretical_prices.append(np.nan)
            implied_vols.append(np.nan)
            price_deviations.append(np.nan)
    
    options_df['theoretical_price'] = theoretical_prices
    options_df['implied_volatility'] = implied_vols
    options_df['price_deviation'] = price_deviations
    
    return options_df


def find_pricing_arbitrage(options_df, min_deviation=0.1):
    """发现定价套利机会"""
    print(f"\n🔍 寻找定价套利机会 (偏差 > {min_deviation*100}%)")
    print("-" * 60)
    
    if options_df.empty:
        return []
    
    arbitrage_ops = []
    
    # 筛选价格偏差较大的期权
    significant_deviations = options_df[
        (options_df['price_deviation'].abs() > min_deviation) &
        (options_df['price_deviation'].notna()) &
        (options_df['vol'] > 0)  # 确保有交易量
    ].copy()
    
    if significant_deviations.empty:
        print("未发现明显的定价套利机会")
        return arbitrage_ops
    
    # 按偏差大小排序
    significant_deviations = significant_deviations.sort_values('price_deviation', key=abs, ascending=False)
    
    print(f"发现 {len(significant_deviations)} 个定价异常的期权:")
    
    for _, row in significant_deviations.head(10).iterrows():
        deviation_pct = row['price_deviation'] * 100
        action = "买入" if row['price_deviation'] < 0 else "卖出"
        
        arbitrage_op = {
            'type': '定价套利',
            'code': row['ts_code'],
            'name': row['name'],
            'action': action,
            'market_price': row['close'],
            'theoretical_price': row['theoretical_price'],
            'deviation': f"{deviation_pct:.2f}%",
            'potential_profit': abs(row['close'] - row['theoretical_price']),
            'volume': row['vol'],
            'days_to_expiry': row['days_to_expiry']
        }
        
        arbitrage_ops.append(arbitrage_op)
        
        print(f"  {action} {row['ts_code']}: 市价 {row['close']:.2f}, 理论 {row['theoretical_price']:.2f}, "
              f"偏差 {deviation_pct:.2f}%, 成交量 {row['vol']:.0f}")
    
    return arbitrage_ops


def find_put_call_parity_arbitrage(options_df, tolerance=0.05):
    """发现期权平价套利机会"""
    print(f"\n🔍 寻找期权平价套利机会 (容差 {tolerance*100}%)")
    print("-" * 60)
    
    arbitrage_ops = []
    
    if options_df.empty:
        return arbitrage_ops
    
    # 按标的和行权价分组
    grouped = options_df.groupby(['underlying', 'exercise_price'])
    parity_violations = []
    
    for (underlying, strike), group in grouped:
        calls = group[group['call_put'] == 'C']
        puts = group[group['call_put'] == 'P']
        
        if calls.empty or puts.empty:
            continue
            
        # 取成交量最大的期权
        call = calls.loc[calls['vol'].idxmax()]
        put = puts.loc[puts['vol'].idxmax()]
        
        if pd.isna(call['underlying_price']) or call['underlying_price'] <= 0:
            continue
        
        # 期权平价关系: C - P = S - K * e^(-r*T)
        # 简化: C - P ≈ S - K
        theoretical_diff = call['underlying_price'] - strike
        actual_diff = call['close'] - put['close']
        
        parity_error = actual_diff - theoretical_diff
        relative_error = abs(parity_error) / max(abs(theoretical_diff), 1)
        
        if relative_error > tolerance:
            parity_violations.append({
                'underlying': underlying,
                'strike': strike,
                'call_code': call['ts_code'],
                'put_code': put['ts_code'],
                'call_price': call['close'],
                'put_price': put['close'],
                'underlying_price': call['underlying_price'],
                'actual_diff': actual_diff,
                'theoretical_diff': theoretical_diff,
                'parity_error': parity_error,
                'relative_error': relative_error,
                'call_vol': call['vol'],
                'put_vol': put['vol'],
                'days_to_expiry': call['days_to_expiry']
            })
    
    if not parity_violations:
        print("未发现期权平价套利机会")
        return arbitrage_ops
    
    # 按偏差大小排序
    parity_violations.sort(key=lambda x: abs(x['parity_error']), reverse=True)
    
    print(f"发现 {len(parity_violations)} 个期权平价偏差:")
    
    for pv in parity_violations[:5]:  # 显示前5个
        if pv['parity_error'] > 0:
            # C - P > S - K, 卖出看涨，买入看跌
            action = "卖出看涨+买入看跌"
        else:
            # C - P < S - K, 买入看涨，卖出看跌  
            action = "买入看涨+卖出看跌"
        
        arbitrage_op = {
            'type': '期权平价套利',
            'underlying': pv['underlying'],
            'strike': pv['strike'],
            'call_code': pv['call_code'],
            'put_code': pv['put_code'],
            'action': action,
            'parity_error': pv['parity_error'],
            'potential_profit': abs(pv['parity_error']),
            'relative_error': f"{pv['relative_error']*100:.2f}%"
        }
        
        arbitrage_ops.append(arbitrage_op)
        
        print(f"  {pv['underlying']} 行权价{pv['strike']}: {action}")
        print(f"    看涨 {pv['call_code']}: {pv['call_price']:.2f}, 看跌 {pv['put_code']}: {pv['put_price']:.2f}")
        print(f"    平价偏差: {pv['parity_error']:.2f}, 相对偏差: {pv['relative_error']*100:.2f}%")
    
    return arbitrage_ops


def find_volatility_arbitrage(options_df, iv_threshold=0.1):
    """发现波动率套利机会"""
    print(f"\n🔍 寻找波动率套利机会")
    print("-" * 60)
    
    arbitrage_ops = []
    
    if options_df.empty:
        return arbitrage_ops
    
    # 按标的分组，分析隐含波动率分布
    volatility_opportunities = []
    
    for underlying in options_df['underlying'].unique():
        if pd.isna(underlying):
            continue
            
        underlying_options = options_df[
            (options_df['underlying'] == underlying) & 
            (options_df['implied_volatility'].notna()) &
            (options_df['implied_volatility'] > 0)
        ]
        
        if len(underlying_options) < 3:
            continue
        
        # 计算波动率统计
        iv_mean = underlying_options['implied_volatility'].mean()
        iv_std = underlying_options['implied_volatility'].std()
        
        if iv_std < iv_threshold:
            continue
        
        # 寻找异常高或低的隐含波动率
        for _, option in underlying_options.iterrows():
            iv_zscore = (option['implied_volatility'] - iv_mean) / iv_std
            
            if abs(iv_zscore) > 2:  # 2倍标准差以外
                volatility_opportunities.append({
                    'code': option['ts_code'],
                    'name': option['name'],
                    'underlying': underlying,
                    'implied_vol': option['implied_volatility'],
                    'iv_zscore': iv_zscore,
                    'iv_mean': iv_mean,
                    'market_price': option['close'],
                    'volume': option['vol'],
                    'days_to_expiry': option['days_to_expiry']
                })
    
    if not volatility_opportunities:
        print("未发现明显的波动率套利机会")
        return arbitrage_ops
    
    # 按Z分数排序
    volatility_opportunities.sort(key=lambda x: abs(x['iv_zscore']), reverse=True)
    
    print(f"发现 {len(volatility_opportunities)} 个波动率异常的期权:")
    
    for vo in volatility_opportunities[:5]:
        if vo['iv_zscore'] > 2:
            action = "卖出期权（波动率高估）"
        else:
            action = "买入期权（波动率低估）"
        
        arbitrage_op = {
            'type': '波动率套利',
            'code': vo['code'],
            'name': vo['name'],
            'action': action,
            'implied_vol': f"{vo['implied_vol']*100:.1f}%",
            'iv_zscore': f"{vo['iv_zscore']:.2f}σ",
            'underlying_avg_iv': f"{vo['iv_mean']*100:.1f}%"
        }
        
        arbitrage_ops.append(arbitrage_op)
        
        print(f"  {action}")
        print(f"    {vo['code']}: IV {vo['implied_vol']*100:.1f}% (Z分数: {vo['iv_zscore']:.2f})")
        print(f"    标的平均IV: {vo['iv_mean']*100:.1f}%, 成交量: {vo['volume']:.0f}")
    
    return arbitrage_ops


def find_calendar_spread_arbitrage(options_df):
    """发现日历价差套利机会"""
    print(f"\n🔍 寻找日历价差套利机会")
    print("-" * 60)
    
    arbitrage_ops = []
    
    if options_df.empty:
        return arbitrage_ops
    
    # 寻找相同标的、相同行权价、不同到期日的期权
    calendar_opportunities = []
    
    # 按标的、期权类型、行权价分组
    grouped = options_df.groupby(['underlying', 'call_put', 'exercise_price'])
    
    for (underlying, option_type, strike), group in grouped:
        if len(group) < 2:
            continue
        
        # 按到期时间排序
        group_sorted = group.sort_values('days_to_expiry')
        
        # 检查相邻到期月份的价格关系
        for i in range(len(group_sorted) - 1):
            near_option = group_sorted.iloc[i]
            far_option = group_sorted.iloc[i + 1]
            
            # 计算时间价值差异
            time_diff = far_option['days_to_expiry'] - near_option['days_to_expiry']
            price_diff = far_option['close'] - near_option['close']
            
            if time_diff <= 0:
                continue
            
            # 时间价值比率分析
            time_value_ratio = price_diff / time_diff if time_diff > 0 else 0
            
            # 寻找异常的时间价值关系
            if (option_type == 'C' and price_diff < 0) or time_value_ratio < 0:
                calendar_opportunities.append({
                    'underlying': underlying,
                    'strike': strike,
                    'option_type': option_type,
                    'near_code': near_option['ts_code'],
                    'far_code': far_option['ts_code'],
                    'near_price': near_option['close'],
                    'far_price': far_option['close'],
                    'near_expiry': near_option['days_to_expiry'],
                    'far_expiry': far_option['days_to_expiry'],
                    'price_diff': price_diff,
                    'time_diff': time_diff,
                    'time_value_ratio': time_value_ratio
                })
    
    if not calendar_opportunities:
        print("未发现日历价差套利机会")
        return arbitrage_ops
    
    print(f"发现 {len(calendar_opportunities)} 个日历价差异常:")
    
    for co in calendar_opportunities[:3]:
        action = "买入远月+卖出近月" if co['price_diff'] < 0 else "卖出远月+买入近月"
        
        arbitrage_op = {
            'type': '日历价差套利',
            'underlying': co['underlying'],
            'strike': co['strike'],
            'near_code': co['near_code'],
            'far_code': co['far_code'],
            'action': action,
            'price_anomaly': co['price_diff'],
            'time_difference': f"{co['time_diff']}天"
        }
        
        arbitrage_ops.append(arbitrage_op)
        
        print(f"  {co['underlying']} 行权价{co['strike']} {co['option_type']}:")
        print(f"    近月 ({co['near_expiry']}天): {co['near_price']:.2f}")
        print(f"    远月 ({co['far_expiry']}天): {co['far_price']:.2f}")
        print(f"    建议: {action}")
    
    return arbitrage_ops


def generate_arbitrage_report(all_arbitrage_ops):
    """生成套利机会报告"""
    print(f"\n📊 期权套利机会汇总报告")
    print("=" * 80)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not all_arbitrage_ops:
        print("\n❌ 未发现明显的套利机会")
        print("建议：")
        print("1. 调整筛选参数（降低偏差阈值）")
        print("2. 增加监控的期权范围")
        print("3. 使用更高频的数据更新")
        return
    
    # 按类型统计
    arbitrage_types = {}
    for op in all_arbitrage_ops:
        op_type = op['type']
        if op_type not in arbitrage_types:
            arbitrage_types[op_type] = []
        arbitrage_types[op_type].append(op)
    
    print(f"\n✅ 发现 {len(all_arbitrage_ops)} 个潜在套利机会:")
    
    for arb_type, ops in arbitrage_types.items():
        print(f"\n🎯 {arb_type}: {len(ops)} 个机会")
        
        for i, op in enumerate(ops[:3], 1):  # 每类显示前3个
            print(f"  {i}. {op.get('code', op.get('near_code', 'N/A'))}")
            if 'action' in op:
                print(f"     操作: {op['action']}")
            if 'potential_profit' in op:
                print(f"     潜在收益: {op['potential_profit']:.2f}")
            if 'deviation' in op:
                print(f"     价格偏差: {op['deviation']}")
    
    print(f"\n⚠️ 风险提示:")
    print("1. 套利机会基于理论分析，实际执行需考虑交易成本")
    print("2. 期权流动性可能影响实际成交价格")
    print("3. 建议结合实时行情验证后再执行")
    print("4. 注意持仓风险和保证金要求")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='期权套利机会发现工具')
    parser.add_argument('-d', '--days', type=int, default=45,
                        help='监控多少天内到期的期权 (默认: 45天)')
    parser.add_argument('--min-deviation', type=float, default=0.08,
                        help='定价套利最小偏差阈值 (默认: 8%)')
    parser.add_argument('--parity-tolerance', type=float, default=0.03,
                        help='期权平价套利容差 (默认: 3%)')
    
    args = parser.parse_args()
    
    print("期权套利机会发现工具")
    print("=" * 60)
    print(f"监控范围: {args.days} 天内到期")
    print(f"定价偏差阈值: {args.min_deviation*100}%")
    print(f"平价套利容差: {args.parity_tolerance*100}%")
    
    try:
        # 初始化数据
        pro = initialize_tushare()
        
        # 获取期权数据
        options_data = get_option_data_with_pricing(pro, args.days)
        
        if options_data.empty:
            print("未获取到有效的期权数据")
            return
        
        print(f"成功加载 {len(options_data)} 个期权数据")
        
        # 发现各类套利机会
        all_arbitrage_ops = []
        
        # 1. 定价套利
        pricing_arbitrage = find_pricing_arbitrage(options_data, args.min_deviation)
        all_arbitrage_ops.extend(pricing_arbitrage)
        
        # 2. 期权平价套利
        parity_arbitrage = find_put_call_parity_arbitrage(options_data, args.parity_tolerance)
        all_arbitrage_ops.extend(parity_arbitrage)
        
        # 3. 波动率套利
        volatility_arbitrage = find_volatility_arbitrage(options_data)
        all_arbitrage_ops.extend(volatility_arbitrage)
        
        # 4. 日历价差套利
        calendar_arbitrage = find_calendar_spread_arbitrage(options_data)
        all_arbitrage_ops.extend(calendar_arbitrage)
        
        # 生成汇总报告
        generate_arbitrage_report(all_arbitrage_ops)
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()