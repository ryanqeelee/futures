"""
数据真实性验证器 - 防止mock数据进入生产系统

CRITICAL: 此模块确保所有金融数据来自真实市场API
绝对禁止任何mock、fake、dummy数据通过验证
"""

import logging
import re
import inspect
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass 
class ValidationResult:
    """验证结果"""
    is_valid: bool
    confidence_score: float  # 0-1, 1为100%可信
    issues: List[str]
    warnings: List[str]
    data_source: Optional[str] = None


class DataAuthenticityValidator:
    """数据真实性验证器"""
    
    # 禁止的mock数据标识符
    FORBIDDEN_PATTERNS = [
        r'mock',
        r'dummy', 
        r'fake',
        r'sample',
        r'test_data',
        r'demo_data',
        r'random\.uniform',
        r'random\.randint',
        r'random\.choice',
        r'np\.random',
        r'Mock\w+',
        r'MOCK_',
        r'TEST_',
        r'DEMO_',
        r'SAMPLE_',
    ]
    
    # 可疑的数据模式
    SUSPICIOUS_PATTERNS = [
        r'OPP_\d{4}',  # 生成的机会ID格式
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}',  # 生成的时间戳格式
    ]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_data_source(self, data: Any, source_info: Dict[str, Any]) -> ValidationResult:
        """验证数据源的真实性"""
        issues = []
        warnings = []
        confidence_score = 1.0
        
        # 检查数据源类型
        source_type = source_info.get('type', '').lower()
        if any(pattern in source_type for pattern in ['mock', 'fake', 'test', 'dummy']):
            issues.append(f"CRITICAL: Forbidden data source type: {source_type}")
            confidence_score = 0.0
            
        # 检查数据源配置
        config = source_info.get('config', {})
        if config.get('generate_random_data'):
            issues.append("CRITICAL: Random data generation is enabled")
            confidence_score = 0.0
            
        # 检查API URL
        api_url = config.get('base_url', '')
        if not api_url or 'localhost' in api_url or '127.0.0.1' in api_url:
            warnings.append("Suspicious API URL - may be local mock server")
            confidence_score *= 0.8
            
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence_score=confidence_score,
            issues=issues,
            warnings=warnings,
            data_source=source_type
        )
    
    def validate_options_data(self, options_df: pd.DataFrame) -> ValidationResult:
        """验证期权数据的真实性"""
        issues = []
        warnings = []
        confidence_score = 1.0
        
        if options_df.empty:
            issues.append("Empty options data")
            return ValidationResult(False, 0.0, issues, warnings)
            
        # 检查数据结构和内容
        required_columns = ['underlying_price', 'exercise_price', 'market_price', 'days_to_expiry']
        missing_columns = [col for col in required_columns if col not in options_df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
            confidence_score *= 0.5
            
        # 检查价格合理性
        if 'market_price' in options_df.columns:
            market_prices = options_df['market_price'].dropna()
            
            # 检查是否有明显生成的价格模式
            if len(market_prices) > 10:
                # 检查价格分布是否过于均匀（可能是随机生成的）
                price_std = market_prices.std()
                price_mean = market_prices.mean()
                cv = price_std / price_mean if price_mean > 0 else 0
                
                if cv < 0.1:  # 变异系数过小，可能是生成数据
                    warnings.append("Price distribution appears artificially uniform")
                    confidence_score *= 0.7
                    
                # 检查是否有重复的价格（真实市场数据应该很少重复）
                unique_ratio = len(market_prices.unique()) / len(market_prices)
                if unique_ratio < 0.8:
                    warnings.append("High ratio of duplicate prices - suspicious")
                    confidence_score *= 0.8
        
        # 检查时间戳
        if 'timestamp' in options_df.columns:
            timestamps = pd.to_datetime(options_df['timestamp'])
            
            # 检查时间戳是否都在同一秒（可能是批量生成的）
            if len(timestamps.unique()) == 1:
                warnings.append("All timestamps identical - possibly generated data")
                confidence_score *= 0.6
                
            # 检查时间戳是否在市场交易时间外
            now = datetime.now()
            future_timestamps = timestamps[timestamps > now]
            if len(future_timestamps) > 0:
                warnings.append("Found future timestamps - data may be simulated")
                confidence_score *= 0.7
        
        # 检查ID模式
        if 'id' in options_df.columns:
            ids = options_df['id'].astype(str)
            for pattern in self.SUSPICIOUS_PATTERNS:
                if ids.str.match(pattern).any():
                    warnings.append(f"Suspicious ID pattern detected: {pattern}")
                    confidence_score *= 0.6
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence_score=confidence_score,
            issues=issues,
            warnings=warnings
        )
    
    def validate_code_for_mock_patterns(self, code_text: str, file_path: str = "") -> ValidationResult:
        """扫描代码中的mock数据模式"""
        issues = []
        warnings = []
        confidence_score = 1.0
        
        code_lower = code_text.lower()
        
        # 检查禁止的模式
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, code_text, re.IGNORECASE):
                issues.append(f"FORBIDDEN: Mock pattern detected in {file_path}: {pattern}")
                confidence_score = 0.0
        
        # 检查硬编码的价格数据
        hardcoded_prices = re.findall(r'\b\d+\.\d{4,}\b', code_text)
        if len(hardcoded_prices) > 5:
            warnings.append(f"Many hardcoded decimal values found - may be mock prices")
            confidence_score *= 0.7
            
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence_score=confidence_score,
            issues=issues,
            warnings=warnings
        )
    
    def validate_runtime_data(self, data: Any, context: str = "") -> ValidationResult:
        """运行时数据验证"""
        issues = []
        warnings = []
        confidence_score = 1.0
        
        # 检查调用栈中是否有mock相关函数
        frame = inspect.currentframe()
        try:
            while frame:
                frame_info = inspect.getframeinfo(frame)
                filename = frame_info.filename.lower()
                function_name = frame.f_code.co_name.lower()
                
                if any(pattern in filename for pattern in ['mock', 'test', 'demo']):
                    if 'test' not in filename:  # 测试文件允许mock
                        issues.append(f"CRITICAL: Data from mock context: {filename}")
                        confidence_score = 0.0
                        
                if any(pattern in function_name for pattern in ['mock', 'generate', 'dummy']):
                    if 'test' not in filename:
                        warnings.append(f"Suspicious function in call stack: {function_name}")
                        confidence_score *= 0.6
                        
                frame = frame.f_back
        finally:
            del frame
            
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence_score=confidence_score,
            issues=issues,
            warnings=warnings
        )


class ProductionDataGuard:
    """生产环境数据守卫"""
    
    def __init__(self):
        self.validator = DataAuthenticityValidator()
        self.logger = logging.getLogger(__name__)
        
    def guard_data_input(self, data: Any, source_info: Dict[str, Any]) -> bool:
        """守卫数据输入 - 在数据进入系统前验证"""
        result = self.validator.validate_data_source(data, source_info)
        
        if not result.is_valid:
            self.logger.error(f"DATA AUTHENTICITY VIOLATION: {result.issues}")
            raise ValueError(f"Mock data detected in production: {result.issues}")
            
        if result.confidence_score < 0.8:
            self.logger.warning(f"Low confidence data detected: {result.warnings}")
            
        return True
        
    def scan_codebase_for_violations(self, root_path: str) -> List[ValidationResult]:
        """扫描代码库寻找违规"""
        import os
        violations = []
        
        for root, dirs, files in os.walk(root_path):
            # 跳过测试、示例、第三方库目录
            if any(skip in root for skip in ['test', 'example', 'demo', '__pycache__', '.git', 'venv', 'site-packages', '.venv', 'node_modules']):
                continue
                
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        result = self.validator.validate_code_for_mock_patterns(content, file_path)
                        if not result.is_valid or result.confidence_score < 0.9:
                            violations.append(result)
                    except Exception as e:
                        self.logger.warning(f"Could not scan {file_path}: {e}")
                        
        return violations


# 全局实例
production_guard = ProductionDataGuard()


def ensure_authentic_data(func):
    """装饰器：确保函数使用真实数据"""
    def wrapper(*args, **kwargs):
        # 运行时检查
        validator = DataAuthenticityValidator()
        result = validator.validate_runtime_data(None, func.__name__)
        
        if not result.is_valid:
            raise RuntimeError(f"MOCK DATA VIOLATION in {func.__name__}: {result.issues}")
            
        return func(*args, **kwargs)
    return wrapper