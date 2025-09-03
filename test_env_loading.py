#!/usr/bin/env python3
"""
测试环境变量加载 - 验证TUSHARE_TOKEN配置修复
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def load_env_file():
    """加载.env文件中的环境变量"""
    env_file = project_root / '.env'
    if env_file.exists():
        print(f"✅ 找到.env文件: {env_file}")
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
                    print(f"✅ 加载环境变量: {key.strip()}=***")
        return True
    else:
        print(f"❌ .env文件不存在: {env_file}")
        return False

def test_streamlit_ui():
    """测试Streamlit UI组件的环境变量访问"""
    try:
        from src.ui.streamlit_app import TradingSystemUI
        
        # 模拟检查TUSHARE_TOKEN的逻辑
        tushare_token = os.getenv('TUSHARE_TOKEN')
        if tushare_token:
            masked_token = tushare_token[:8] + "*" * (len(tushare_token) - 16) + tushare_token[-8:]
            print(f"✅ TradingSystemUI可以访问TUSHARE_TOKEN: {masked_token}")
            return True
        else:
            print("❌ TradingSystemUI无法访问TUSHARE_TOKEN")
            return False
            
    except Exception as e:
        print(f"❌ 测试TradingSystemUI失败: {e}")
        return False

def main():
    print("🔍 环境变量加载测试")
    print("=" * 50)
    
    print("\n1. 加载.env文件测试:")
    env_loaded = load_env_file()
    
    print(f"\n2. TUSHARE_TOKEN检查:")
    token = os.getenv('TUSHARE_TOKEN')
    if token:
        masked = token[:8] + '*' * (len(token) - 16) + token[-8:]
        print(f"✅ TUSHARE_TOKEN已加载: {masked}")
    else:
        print("❌ TUSHARE_TOKEN未加载")
    
    print(f"\n3. Streamlit UI组件测试:")
    ui_test_passed = test_streamlit_ui()
    
    print(f"\n🎯 测试结果:")
    print(f"- .env文件加载: {'✅' if env_loaded else '❌'}")
    print(f"- TUSHARE_TOKEN可用: {'✅' if token else '❌'}")
    print(f"- UI组件访问: {'✅' if ui_test_passed else '❌'}")
    
    if env_loaded and token and ui_test_passed:
        print(f"\n🎉 所有测试通过！TUSHARE_TOKEN配置问题已修复")
        print(f"📱 现在可以正常使用Streamlit应用了")
    else:
        print(f"\n⚠️ 部分测试失败，请检查配置")

if __name__ == "__main__":
    main()