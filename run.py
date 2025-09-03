"""
期权套利扫描工具 - 命令行启动器
Option Arbitrage Scanner - Command Line Launcher

提供命令行方式启动应用的入口
"""

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    """命令行启动器"""
    parser = argparse.ArgumentParser(description="期权套利扫描工具")
    parser.add_argument(
        "--mode", 
        choices=["web", "scan"], 
        default="web",
        help="运行模式: web(Web界面) 或 scan(命令行扫描)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8501,
        help="Web服务端口 (默认: 8501)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "web":
        # 启动Streamlit Web界面
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "app.py", "--server.port", str(args.port)
        ]
        subprocess.run(cmd)
    
    elif args.mode == "scan":
        # 命令行扫描模式
        print("命令行扫描模式开发中...")
        # TODO: 实现命令行扫描功能

if __name__ == "__main__":
    main()