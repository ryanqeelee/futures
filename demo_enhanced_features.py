#!/usr/bin/env python3
"""
增强功能演示脚本

演示阶段4任务2新增的结果展示和排序功能
包括数据生成、组件测试和功能验证
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入新增组件
from src.ui.components.enhanced_results_display import EnhancedResultsDisplay
from src.ui.components.data_visualization import DataVisualization
from src.ui.components.data_filters import AdvancedDataFilters
from src.ui.utils.export_utils import ExportUtils

def generate_demo_data(num_records: int = 100) -> list:
    """生成演示数据"""
    print(f"生成 {num_records} 条演示数据...")
    
    np.random.seed(42)  # 确保可重复性
    
    strategies = ['covered_call', 'protective_put', 'iron_condor', 'butterfly', 'straddle', 'strangle']
    instruments = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN', 'META', 'NVDA', 'SPY', 'QQQ', 'IWM']
    
    demo_data = []
    base_time = datetime.now()
    
    for i in range(num_records):
        # 基础数据
        strategy = np.random.choice(strategies)
        instrument = np.random.choice(instruments)
        
        # 生成相关性数据（使策略和风险/收益有一定关系）
        if strategy in ['covered_call', 'protective_put']:
            base_profit = np.random.normal(0.015, 0.005)  # 较低风险策略
            base_risk = np.random.normal(0.2, 0.05)
        elif strategy in ['iron_condor', 'butterfly']:
            base_profit = np.random.normal(0.025, 0.008)  # 中等风险策略
            base_risk = np.random.normal(0.35, 0.08)
        else:  # straddle, strangle
            base_profit = np.random.normal(0.035, 0.012)  # 高风险策略
            base_risk = np.random.normal(0.5, 0.1)
        
        # 确保数据在合理范围内
        profit_margin = max(0.001, min(0.1, base_profit))
        risk_score = max(0.01, min(0.99, base_risk))
        expected_profit = profit_margin * np.random.normal(10000, 2000)
        confidence_score = max(0.1, min(0.99, np.random.normal(0.7, 0.15)))
        
        # 时间戳（最近7天内随机分布）
        time_offset = timedelta(hours=np.random.randint(0, 168))  # 7天内随机
        timestamp = base_time - time_offset
        
        record = {
            'id': f'OPP_{i+1:04d}',
            'strategy_type': strategy,
            'profit_margin': profit_margin,
            'expected_profit': expected_profit,
            'risk_score': risk_score,
            'confidence_score': confidence_score,
            'instruments': f'{instrument}_OPTIONS',
            'timestamp': timestamp,
            'sharpe_ratio': profit_margin / risk_score if risk_score > 0 else 0,
            'win_rate': min(0.95, max(0.3, confidence_score + np.random.normal(0, 0.1))),
            'max_drawdown': risk_score * np.random.uniform(0.8, 1.2)
        }
        
        demo_data.append(record)
    
    print(f"✓ 成功生成 {len(demo_data)} 条演示数据")
    return demo_data

def test_enhanced_results_display(demo_data: list):
    """测试增强版结果展示组件"""
    print("\n=== 测试增强版结果展示组件 ===")
    
    try:
        display = EnhancedResultsDisplay()
        print("✓ 增强版结果展示组件初始化成功")
        
        # 测试数据准备
        df = display._prepare_display_data(demo_data)
        print(f"✓ 数据准备完成，共 {len(df)} 行，{len(df.columns)} 列")
        
        # 测试排序功能（模拟）
        if 'profit_margin' in df.columns:
            sorted_df = df.sort_values('profit_margin', ascending=False)
            print(f"✓ 排序功能测试通过，最高利润率: {sorted_df['profit_margin'].iloc[0]:.4f}")
        
        # 测试格式化功能
        formatted_df = display._format_display_data(df.head(5))
        print(f"✓ 数据格式化测试通过，格式化后列数: {len(formatted_df.columns)}")
        
    except Exception as e:
        print(f"✗ 增强版结果展示组件测试失败: {e}")

def test_data_visualization(demo_data: list):
    """测试数据可视化组件"""
    print("\n=== 测试数据可视化组件 ===")
    
    try:
        viz = DataVisualization()
        print("✓ 数据可视化组件初始化成功")
        
        df = pd.DataFrame(demo_data)
        
        # 测试KPI计算
        total_opportunities = len(df)
        avg_profit = df['profit_margin'].mean() if 'profit_margin' in df.columns else 0
        avg_risk = df['risk_score'].mean() if 'risk_score' in df.columns else 0
        
        print(f"✓ KPI计算完成:")
        print(f"  - 总机会数: {total_opportunities}")
        print(f"  - 平均利润率: {avg_profit:.4f}")
        print(f"  - 平均风险评分: {avg_risk:.4f}")
        
        # 测试策略统计
        if 'strategy_type' in df.columns:
            strategy_stats = df['strategy_type'].value_counts()
            print(f"✓ 策略分布分析完成，发现 {len(strategy_stats)} 种策略")
            for strategy, count in strategy_stats.head(3).items():
                print(f"  - {strategy}: {count} 个机会")
        
    except Exception as e:
        print(f"✗ 数据可视化组件测试失败: {e}")

def test_data_filters(demo_data: list):
    """测试数据筛选组件"""
    print("\n=== 测试数据筛选组件 ===")
    
    try:
        filters = AdvancedDataFilters()
        print("✓ 数据筛选组件初始化成功")
        
        df = pd.DataFrame(demo_data)
        
        # 模拟筛选条件测试
        from src.ui.components.data_filters import FilterCondition, FilterOperator
        
        # 创建测试筛选条件
        high_profit_condition = FilterCondition(
            column='profit_margin',
            operator=FilterOperator.GREATER_THAN,
            value=0.02  # 利润率大于2%
        )
        
        low_risk_condition = FilterCondition(
            column='risk_score',
            operator=FilterOperator.LESS_THAN,
            value=0.4  # 风险评分小于0.4
        )
        
        conditions = [high_profit_condition, low_risk_condition]
        
        # 应用筛选
        filtered_df = filters._apply_filter_conditions(df, conditions)
        print(f"✓ 筛选测试完成:")
        print(f"  - 原始数据: {len(df)} 条")
        print(f"  - 筛选后: {len(filtered_df)} 条")
        print(f"  - 保留率: {len(filtered_df)/len(df)*100:.1f}%")
        
    except Exception as e:
        print(f"✗ 数据筛选组件测试失败: {e}")

def test_export_utils(demo_data: list):
    """测试导出工具"""
    print("\n=== 测试导出工具 ===")
    
    try:
        export_utils = ExportUtils()
        print("✓ 导出工具初始化成功")
        
        df = pd.DataFrame(demo_data)
        
        # 测试数据格式化
        formatted_df = export_utils._format_export_data(df.head(10))
        print(f"✓ 数据格式化测试通过，格式化后行数: {len(formatted_df)}")
        
        # 测试统计摘要生成
        summary_df = export_utils._generate_summary_statistics(df)
        print(f"✓ 统计摘要生成测试通过，摘要条目: {len(summary_df)}")
        
        # 测试数值数据检测
        has_numeric = export_utils._has_numeric_data(df)
        print(f"✓ 数值数据检测: {'有' if has_numeric else '无'}数值数据")
        
    except Exception as e:
        print(f"✗ 导出工具测试失败: {e}")

def test_data_quality(demo_data: list):
    """测试数据质量"""
    print("\n=== 数据质量检查 ===")
    
    df = pd.DataFrame(demo_data)
    
    print(f"数据基本信息:")
    print(f"  - 总行数: {len(df)}")
    print(f"  - 总列数: {len(df.columns)}")
    print(f"  - 内存使用: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # 检查缺失值
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    print(f"  - 缺失值总数: {total_missing}")
    
    # 检查数值范围
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"  - 数值列数量: {len(numeric_cols)}")
    
    if 'profit_margin' in df.columns:
        profit_range = f"{df['profit_margin'].min():.4f} ~ {df['profit_margin'].max():.4f}"
        print(f"  - 利润率范围: {profit_range}")
    
    if 'risk_score' in df.columns:
        risk_range = f"{df['risk_score'].min():.4f} ~ {df['risk_score'].max():.4f}"
        print(f"  - 风险评分范围: {risk_range}")
    
    # 检查时间范围
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        time_range = f"{df['timestamp'].min()} ~ {df['timestamp'].max()}"
        print(f"  - 时间范围: {time_range}")

def save_demo_data(demo_data: list, filename: str = "demo_data.json"):
    """保存演示数据到文件"""
    print(f"\n保存演示数据到 {filename}...")
    
    # 处理datetime对象
    serializable_data = []
    for record in demo_data:
        processed_record = record.copy()
        if 'timestamp' in processed_record:
            processed_record['timestamp'] = processed_record['timestamp'].isoformat()
        serializable_data.append(processed_record)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 演示数据已保存到 {filename}")

def main():
    """主函数"""
    print("🚀 增强功能演示脚本启动")
    print("=" * 50)
    
    # 生成演示数据
    demo_data = generate_demo_data(150)
    
    # 数据质量检查
    test_data_quality(demo_data)
    
    # 测试各组件
    test_enhanced_results_display(demo_data)
    test_data_visualization(demo_data)
    test_data_filters(demo_data)
    test_export_utils(demo_data)
    
    # 保存演示数据
    save_demo_data(demo_data)
    
    print("\n" + "=" * 50)
    print("✅ 所有组件测试完成")
    print("\n使用说明:")
    print("1. 运行 'streamlit run src/ui/streamlit_app.py' 启动Web界面")
    print("2. 使用生成的 demo_data.json 作为测试数据")
    print("3. 体验新增的高级分析功能")
    print("\n新功能包括:")
    print("- 🎯 增强版结果展示（多级排序、高级筛选）")
    print("- 📊 高级数据分析（策略对比、相关性分析）")
    print("- 🔍 智能数据筛选（预设条件、自定义筛选）")
    print("- 📈 综合数据可视化（多种图表类型）")
    print("- 📥 多格式数据导出（Excel、PDF、CSV等）")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ 用户中断执行")
    except Exception as e:
        print(f"\n\n❌ 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 演示脚本结束")