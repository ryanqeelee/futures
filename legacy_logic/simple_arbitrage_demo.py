#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版期权套利发现演示
展示核心套利概念和实现思路
"""

import os
import sys
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


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


def get_option_sample_data(pro, max_days=45):
    """获取期权样本数据"""
    print("获取期权样本数据...")
    
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
            print(f"未找到{max_days}天内到期的期权")
            return pd.DataFrame()
        
        # 获取最新行情数据
        for days_back in range(1, 5):
            trade_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
            try:
                daily_data = pro.opt_daily(trade_date=trade_date)
                if not daily_data.empty:
                    print(f"获取到 {trade_date} 的行情数据")
                    break
            except:
                continue
        
        if daily_data.empty:
            print("未能获取行情数据")
            return pd.DataFrame()
        
        # 合并数据
        full_data = near_options.merge(
            daily_data[['ts_code', 'close', 'vol', 'oi']], 
            on='ts_code', 
            how='inner'
        )
        
        # 过滤有效数据
        full_data = full_data[
            (full_data['close'] > 0) & 
            (full_data['vol'] > 0) &
            (full_data['exercise_price'] > 0)
        ].copy()
        
        print(f"获得 {len(full_data)} 个有效期权数据")
        return full_data
        
    except Exception as e:
        print(f"获取数据失败: {e}")
        return pd.DataFrame()


def find_simple_pricing_anomalies(options_df, deviation_threshold=0.15):
    """发现简单的定价异常"""
    print(f"\n🔍 寻找定价异常 (阈值: {deviation_threshold*100}%)")
    print("-" * 50)
    
    if options_df.empty:
        return []
    
    anomalies = []
    
    # 简单的相对价值分析
    for underlying in options_df['ts_code'].str.extract(r'([A-Z]+\d{4})')[0].unique():
        if pd.isna(underlying):
            continue
            
        # 获取同一标的的期权
        underlying_options = options_df[
            options_df['ts_code'].str.contains(underlying, na=False)
        ].copy()
        
        if len(underlying_options) < 5:
            continue
        
        # 按期权类型分析
        for option_type in ['C', 'P']:
            type_options = underlying_options[
                underlying_options['call_put'] == option_type
            ].copy()
            
            if len(type_options) < 3:
                continue
            
            # 计算价格相对于行权价的比例
            type_options['price_ratio'] = type_options['close'] / type_options['exercise_price']
            
            # 找到异常高或低的价格比例
            mean_ratio = type_options['price_ratio'].mean()
            std_ratio = type_options['price_ratio'].std()
            
            if std_ratio == 0:
                continue
            
            for _, option in type_options.iterrows():
                z_score = abs((option['price_ratio'] - mean_ratio) / std_ratio)
                
                if z_score > 2:  # 超过2个标准差
                    anomalies.append({
                        'code': option['ts_code'],
                        'name': option['name'],
                        'price': option['close'],
                        'strike': option['exercise_price'],
                        'type': '认购' if option_type == 'C' else '认沽',
                        'price_ratio': option['price_ratio'],
                        'z_score': z_score,
                        'volume': option['vol'],
                        'days_to_expiry': option['days_to_expiry'],
                        'anomaly_type': '价格异常高' if option['price_ratio'] > mean_ratio else '价格异常低'
                    })
    
    # 按异常程度排序
    anomalies.sort(key=lambda x: x['z_score'], reverse=True)
    
    if not anomalies:
        print("未发现明显的定价异常")
        return anomalies
    
    print(f"发现 {len(anomalies)} 个定价异常:")
    
    for i, anomaly in enumerate(anomalies[:5], 1):
        print(f"{i}. {anomaly['code']} ({anomaly['type']})")
        print(f"   价格: {anomaly['price']:.2f}, 行权价: {anomaly['strike']:.0f}")
        print(f"   {anomaly['anomaly_type']} (Z分数: {anomaly['z_score']:.2f})")
        print(f"   成交量: {anomaly['volume']:.0f}, 剩余天数: {anomaly['days_to_expiry']}")
    
    return anomalies


def find_put_call_parity_opportunities(options_df, tolerance=0.05):
    """寻找期权平价机会（简化版）"""
    print(f"\n🔍 寻找期权平价机会 (容差: {tolerance*100}%)")
    print("-" * 50)
    
    if options_df.empty:
        return []
    
    parity_ops = []
    
    # 按标的分组
    options_df['underlying'] = options_df['ts_code'].str.extract(r'([A-Z]+\d{4})')[0]
    
    for underlying in options_df['underlying'].unique():
        if pd.isna(underlying):
            continue
        
        underlying_options = options_df[options_df['underlying'] == underlying].copy()
        
        # 按行权价分组寻找配对
        for strike in underlying_options['exercise_price'].unique():
            if pd.isna(strike):
                continue
            
            strike_options = underlying_options[underlying_options['exercise_price'] == strike]
            
            calls = strike_options[strike_options['call_put'] == 'C']
            puts = strike_options[strike_options['call_put'] == 'P']
            
            if calls.empty or puts.empty:
                continue
            
            # 取成交量最大的期权
            best_call = calls.loc[calls['vol'].idxmax()]
            best_put = puts.loc[puts['vol'].idxmax()]
            
            # 估算标的价格（使用行权价作为近似）
            estimated_underlying = strike  # 简化假设
            
            # 期权平价关系检查: C - P ≈ S - K (忽略利率和时间价值)
            theoretical_diff = estimated_underlying - strike
            actual_diff = best_call['close'] - best_put['close']
            
            parity_error = abs(actual_diff - theoretical_diff)
            relative_error = parity_error / max(abs(theoretical_diff), 1)
            
            if relative_error > tolerance:
                parity_ops.append({
                    'underlying': underlying,
                    'strike': strike,
                    'call_code': best_call['ts_code'],
                    'put_code': best_put['ts_code'],
                    'call_price': best_call['close'],
                    'put_price': best_put['close'],
                    'actual_diff': actual_diff,
                    'theoretical_diff': theoretical_diff,
                    'parity_error': parity_error,
                    'relative_error': relative_error,
                    'opportunity': '买入看涨卖出看跌' if actual_diff < theoretical_diff else '卖出看涨买入看跌'
                })
    
    if not parity_ops:
        print("未发现期权平价机会")
        return parity_ops
    
    # 按错误大小排序
    parity_ops.sort(key=lambda x: x['parity_error'], reverse=True)
    
    print(f"发现 {len(parity_ops)} 个平价机会:")
    
    for i, op in enumerate(parity_ops[:3], 1):
        print(f"{i}. {op['underlying']} 行权价 {op['strike']:.0f}")
        print(f"   看涨: {op['call_price']:.2f} vs 看跌: {op['put_price']:.2f}")
        print(f"   平价偏差: {op['parity_error']:.2f} ({op['relative_error']*100:.1f}%)")
        print(f"   建议: {op['opportunity']}")
    
    return parity_ops


def find_time_value_opportunities(options_df):
    """寻找时间价值套利机会"""
    print(f"\n🔍 寻找时间价值机会")
    print("-" * 50)
    
    if options_df.empty:
        return []
    
    time_ops = []
    
    # 按标的和期权类型分组
    options_df['underlying'] = options_df['ts_code'].str.extract(r'([A-Z]+\d{4})')[0]
    
    for underlying in options_df['underlying'].unique():
        if pd.isna(underlying):
            continue
            
        underlying_options = options_df[options_df['underlying'] == underlying]
        
        for option_type in ['C', 'P']:
            type_options = underlying_options[underlying_options['call_put'] == option_type]
            
            if len(type_options) < 2:
                continue
            
            # 按到期时间排序
            type_options_sorted = type_options.sort_values('days_to_expiry')
            
            # 寻找相似行权价但不同到期时间的期权
            for i in range(len(type_options_sorted)):
                for j in range(i+1, len(type_options_sorted)):
                    option1 = type_options_sorted.iloc[i]
                    option2 = type_options_sorted.iloc[j]
                    
                    # 行权价差异不能太大
                    strike_diff = abs(option1['exercise_price'] - option2['exercise_price'])
                    avg_strike = (option1['exercise_price'] + option2['exercise_price']) / 2
                    
                    if strike_diff / avg_strike > 0.1:  # 行权价差异超过10%
                        continue
                    
                    # 时间差异分析
                    time_diff = option2['days_to_expiry'] - option1['days_to_expiry']
                    price_diff = option2['close'] - option1['close']
                    
                    if time_diff <= 0:
                        continue
                    
                    # 计算每天的时间价值
                    time_value_per_day = price_diff / time_diff
                    
                    # 寻找异常的时间价值
                    if time_value_per_day < 0 or time_value_per_day > option1['close'] * 0.1:
                        time_ops.append({
                            'underlying': underlying,
                            'near_code': option1['ts_code'],
                            'far_code': option2['ts_code'],
                            'near_price': option1['close'],
                            'far_price': option2['close'],
                            'near_days': option1['days_to_expiry'],
                            'far_days': option2['days_to_expiry'],
                            'time_diff': time_diff,
                            'price_diff': price_diff,
                            'time_value_per_day': time_value_per_day,
                            'opportunity': '异常时间价值' if time_value_per_day < 0 else '时间价值过高'
                        })
    
    if not time_ops:
        print("未发现明显的时间价值机会")
        return time_ops
    
    # 按异常程度排序
    time_ops.sort(key=lambda x: abs(x['time_value_per_day']), reverse=True)
    
    print(f"发现 {len(time_ops)} 个时间价值机会:")
    
    for i, op in enumerate(time_ops[:3], 1):
        print(f"{i}. {op['underlying']}")
        print(f"   近月 ({op['near_days']}天): {op['near_price']:.2f}")
        print(f"   远月 ({op['far_days']}天): {op['far_price']:.2f}")
        print(f"   每日时间价值: {op['time_value_per_day']:.3f}")
        print(f"   {op['opportunity']}")
    
    return time_ops


def generate_demo_report(pricing_anomalies, parity_ops, time_ops):
    """生成演示报告"""
    print(f"\n📊 期权套利机会演示报告")
    print("=" * 70)
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_opportunities = len(pricing_anomalies) + len(parity_ops) + len(time_ops)
    
    if total_opportunities == 0:
        print("\n❌ 当前未发现明显的套利机会")
        print("\n可能的原因:")
        print("1. 市场效率较高，套利机会稀少")
        print("2. 筛选条件过于严格")
        print("3. 需要更高频的数据或更精确的定价模型")
        return
    
    print(f"\n✅ 发现 {total_opportunities} 个潜在机会:")
    print(f"   定价异常: {len(pricing_anomalies)} 个")
    print(f"   期权平价: {len(parity_ops)} 个")
    print(f"   时间价值: {len(time_ops)} 个")
    
    print(f"\n💡 套利策略建议:")
    
    if pricing_anomalies:
        print(f"\n1. 定价异常套利:")
        print("   - 买入被低估的期权")
        print("   - 卖出被高估的期权")
        print("   - 注意流动性和交易成本")
    
    if parity_ops:
        print(f"\n2. 期权平价套利:")
        print("   - 构建delta中性组合")
        print("   - 利用看涨看跌期权的定价偏差")
        print("   - 适合机构投资者")
    
    if time_ops:
        print(f"\n3. 时间价值套利:")
        print("   - 日历价差策略")
        print("   - 利用时间衰减的不一致性")
        print("   - 适合波动率交易")
    
    print(f"\n⚠️ 风险提示:")
    print("1. 这是基于历史数据的理论分析")
    print("2. 实际交易需考虑:")
    print("   - 交易成本和滑点")
    print("   - 保证金要求")
    print("   - 流动性风险")
    print("   - 市场风险")
    print("3. 建议在模拟环境中测试策略")


def main():
    """主函数"""
    print("期权套利机会发现演示")
    print("=" * 50)
    
    try:
        # 初始化
        pro = initialize_tushare()
        
        # 获取样本数据
        options_data = get_option_sample_data(pro, max_days=45)
        
        if options_data.empty:
            print("未获取到有效数据，演示结束")
            return
        
        print(f"成功加载 {len(options_data)} 个期权样本")
        
        # 寻找各类套利机会
        pricing_anomalies = find_simple_pricing_anomalies(options_data)
        parity_ops = find_put_call_parity_opportunities(options_data)
        time_ops = find_time_value_opportunities(options_data)
        
        # 生成报告
        generate_demo_report(pricing_anomalies, parity_ops, time_ops)
        
        print(f"\n🎯 下一步建议:")
        print("1. 使用 option_arbitrage_scanner.py 进行更精确的分析")
        print("2. 结合实时数据验证机会")
        print("3. 开发自动化监控系统")
        print("4. 回测历史套利策略表现")
        
    except Exception as e:
        print(f"演示程序出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()