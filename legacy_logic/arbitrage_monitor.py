#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期权套利机会实时监控系统
定期扫描和报警套利机会
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import pandas as pd
from simple_arbitrage_demo import (
    initialize_tushare, 
    get_option_sample_data,
    find_simple_pricing_anomalies,
    find_put_call_parity_opportunities,
    find_time_value_opportunities
)


class ArbitrageMonitor:
    """套利机会监控类"""
    
    def __init__(self, config_file='arbitrage_config.json'):
        self.config = self.load_config(config_file)
        self.setup_logging()
        self.pro = initialize_tushare()
        self.last_opportunities = {}
        
    def load_config(self, config_file):
        """加载配置文件"""
        default_config = {
            "monitoring": {
                "interval_minutes": 30,
                "max_days_to_expiry": 45,
                "pricing_deviation_threshold": 0.12,
                "parity_tolerance": 0.05,
                "min_volume": 1
            },
            "alerts": {
                "enable_console": True,
                "enable_file": True,
                "enable_email": False,
                "log_file": "arbitrage_alerts.log",
                "alert_threshold": 0.15
            },
            "filters": {
                "exclude_low_volume": True,
                "min_volume": 5,
                "focus_exchanges": ["GFE", "DCE", "CZCE", "SHFE"],
                "exclude_deep_otm": True,
                "max_moneyness": 2.0
            }
        }
        
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"配置文件读取失败，使用默认配置: {e}")
        else:
            # 创建默认配置文件
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            print(f"已创建默认配置文件: {config_file}")
        
        return default_config
    
    def setup_logging(self):
        """设置日志"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        if self.config['alerts']['enable_file']:
            logging.basicConfig(
                level=logging.INFO,
                format=log_format,
                handlers=[
                    logging.FileHandler(self.config['alerts']['log_file'], encoding='utf-8'),
                    logging.StreamHandler(sys.stdout) if self.config['alerts']['enable_console'] else logging.NullHandler()
                ]
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format=log_format,
                handlers=[logging.StreamHandler(sys.stdout)]
            )
        
        self.logger = logging.getLogger(__name__)
    
    def filter_options_data(self, options_df):
        """根据配置过滤期权数据"""
        if options_df.empty:
            return options_df
        
        filtered_df = options_df.copy()
        
        # 成交量过滤
        if self.config['filters']['exclude_low_volume']:
            min_vol = self.config['filters']['min_volume']
            filtered_df = filtered_df[filtered_df['vol'] >= min_vol]
        
        # 交易所过滤
        focus_exchanges = self.config['filters']['focus_exchanges']
        if focus_exchanges:
            exchange_pattern = '|'.join(focus_exchanges)
            filtered_df = filtered_df[
                filtered_df['ts_code'].str.contains(exchange_pattern, na=False)
            ]
        
        # 排除深度虚值期权
        if self.config['filters']['exclude_deep_otm']:
            max_moneyness = self.config['filters']['max_moneyness']
            # 简化的虚实值判断
            filtered_df = filtered_df[
                (filtered_df['close'] / filtered_df['exercise_price'] < max_moneyness) &
                (filtered_df['close'] / filtered_df['exercise_price'] > 1/max_moneyness)
            ]
        
        return filtered_df
    
    def scan_arbitrage_opportunities(self):
        """扫描套利机会"""
        try:
            # 获取数据
            options_data = get_option_sample_data(
                self.pro, 
                max_days=self.config['monitoring']['max_days_to_expiry']
            )
            
            if options_data.empty:
                self.logger.warning("未获取到期权数据")
                return {}
            
            # 过滤数据
            filtered_data = self.filter_options_data(options_data)
            self.logger.info(f"扫描 {len(filtered_data)} 个期权合约")
            
            # 发现套利机会
            opportunities = {}
            
            # 定价异常
            pricing_anomalies = find_simple_pricing_anomalies(
                filtered_data, 
                deviation_threshold=self.config['monitoring']['pricing_deviation_threshold']
            )
            if pricing_anomalies:
                opportunities['pricing_anomalies'] = pricing_anomalies
            
            # 期权平价
            parity_ops = find_put_call_parity_opportunities(
                filtered_data,
                tolerance=self.config['monitoring']['parity_tolerance']
            )
            if parity_ops:
                opportunities['parity_opportunities'] = parity_ops
            
            # 时间价值
            time_ops = find_time_value_opportunities(filtered_data)
            if time_ops:
                opportunities['time_value_opportunities'] = time_ops
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"扫描套利机会时出错: {e}")
            return {}
    
    def analyze_opportunity_changes(self, current_ops):
        """分析套利机会变化"""
        changes = {
            'new_opportunities': [],
            'disappeared_opportunities': [],
            'changed_opportunities': []
        }
        
        # 比较与上次扫描的差异
        current_codes = set()
        for category, ops in current_ops.items():
            for op in ops:
                if 'code' in op:
                    current_codes.add(op['code'])
                elif 'call_code' in op and 'put_code' in op:
                    current_codes.add(f"{op['call_code']}-{op['put_code']}")
        
        last_codes = set(self.last_opportunities.keys())
        
        # 新出现的机会
        new_codes = current_codes - last_codes
        if new_codes:
            changes['new_opportunities'] = list(new_codes)
        
        # 消失的机会
        disappeared_codes = last_codes - current_codes
        if disappeared_codes:
            changes['disappeared_opportunities'] = list(disappeared_codes)
        
        # 更新历史记录
        self.last_opportunities = {code: datetime.now() for code in current_codes}
        
        return changes
    
    def send_alert(self, opportunities, changes):
        """发送警报"""
        total_ops = sum(len(ops) for ops in opportunities.values())
        
        if total_ops == 0 and not changes['disappeared_opportunities']:
            return
        
        # 构建警报消息
        alert_msg = f"\n{'='*60}\n"
        alert_msg += f"期权套利机会监控报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        alert_msg += f"{'='*60}\n"
        
        if total_ops > 0:
            alert_msg += f"\n🎯 发现 {total_ops} 个套利机会:\n"
            
            for category, ops in opportunities.items():
                category_name = {
                    'pricing_anomalies': '定价异常',
                    'parity_opportunities': '期权平价',
                    'time_value_opportunities': '时间价值'
                }.get(category, category)
                
                alert_msg += f"\n📍 {category_name}: {len(ops)} 个\n"
                
                for i, op in enumerate(ops[:3], 1):  # 只显示前3个
                    if 'code' in op:
                        alert_msg += f"  {i}. {op['code']}: {op.get('anomaly_type', op.get('opportunity', '套利机会'))}\n"
                        if 'price' in op and 'z_score' in op:
                            alert_msg += f"     价格: {op['price']:.2f}, 异常度: {op['z_score']:.2f}σ\n"
                    elif 'underlying' in op:
                        alert_msg += f"  {i}. {op['underlying']} 行权价{op.get('strike', 'N/A')}: {op.get('opportunity', '套利机会')}\n"
        
        # 变化信息
        if changes['new_opportunities']:
            alert_msg += f"\n🆕 新发现机会: {len(changes['new_opportunities'])} 个\n"
            for code in changes['new_opportunities'][:5]:
                alert_msg += f"  • {code}\n"
        
        if changes['disappeared_opportunities']:
            alert_msg += f"\n❌ 消失机会: {len(changes['disappeared_opportunities'])} 个\n"
        
        # 发送警报
        if self.config['alerts']['enable_console']:
            print(alert_msg)
        
        if self.config['alerts']['enable_file']:
            self.logger.info(alert_msg.replace('\n', ' | '))
        
        # TODO: 实现邮件警报
        if self.config['alerts']['enable_email']:
            self.send_email_alert(alert_msg)
    
    def send_email_alert(self, message):
        """发送邮件警报（待实现）"""
        # 这里可以实现邮件发送功能
        self.logger.info("邮件警报功能待实现")
    
    def run_continuous_monitoring(self):
        """运行连续监控"""
        interval_minutes = self.config['monitoring']['interval_minutes']
        
        self.logger.info(f"开始期权套利机会监控，扫描间隔: {interval_minutes} 分钟")
        
        try:
            while True:
                self.logger.info("开始新一轮扫描...")
                
                # 扫描套利机会
                opportunities = self.scan_arbitrage_opportunities()
                
                # 分析变化
                changes = self.analyze_opportunity_changes(opportunities)
                
                # 发送警报
                self.send_alert(opportunities, changes)
                
                # 等待下一次扫描
                sleep_seconds = interval_minutes * 60
                self.logger.info(f"等待 {interval_minutes} 分钟后进行下次扫描...")
                time.sleep(sleep_seconds)
                
        except KeyboardInterrupt:
            self.logger.info("监控已停止")
        except Exception as e:
            self.logger.error(f"监控系统出错: {e}")
    
    def run_single_scan(self):
        """运行单次扫描"""
        self.logger.info("执行单次套利机会扫描")
        
        opportunities = self.scan_arbitrage_opportunities()
        changes = {'new_opportunities': [], 'disappeared_opportunities': [], 'changed_opportunities': []}
        
        self.send_alert(opportunities, changes)
        
        return opportunities


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='期权套利机会监控系统')
    parser.add_argument('-c', '--config', type=str, default='arbitrage_config.json',
                        help='配置文件路径')
    parser.add_argument('--single', action='store_true',
                        help='执行单次扫描而非连续监控')
    parser.add_argument('--interval', type=int,
                        help='监控间隔（分钟），覆盖配置文件设置')
    
    args = parser.parse_args()
    
    try:
        # 初始化监控器
        monitor = ArbitrageMonitor(args.config)
        
        # 覆盖间隔设置
        if args.interval:
            monitor.config['monitoring']['interval_minutes'] = args.interval
        
        if args.single:
            # 单次扫描
            opportunities = monitor.run_single_scan()
            total_ops = sum(len(ops) for ops in opportunities.values())
            print(f"\n扫描完成，发现 {total_ops} 个潜在套利机会")
        else:
            # 连续监控
            monitor.run_continuous_monitoring()
    
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()