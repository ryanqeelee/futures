"""
Data quality validation and monitoring module for data adapters.
Provides comprehensive data validation, anomaly detection, and quality metrics.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats

from ..strategies.base import OptionData
from .base import DataQuality


class ValidationRule(str, Enum):
    """Data validation rule types."""
    PRICE_RANGE = "price_range"
    VOLUME_RANGE = "volume_range"
    IV_RANGE = "implied_volatility_range"
    PRICE_CONSISTENCY = "price_consistency"
    GREEK_CONSISTENCY = "greek_consistency"
    TIME_CONSISTENCY = "time_consistency"
    MISSING_DATA = "missing_data"
    DUPLICATE_DATA = "duplicate_data"


class ValidationSeverity(str, Enum):
    """Validation error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationError:
    """Data validation error details."""
    rule: ValidationRule
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    record_id: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class QualityMetrics:
    """Data quality metrics."""
    completeness_score: float  # 0-1
    accuracy_score: float      # 0-1
    consistency_score: float   # 0-1
    timeliness_score: float    # 0-1
    validity_score: float      # 0-1
    overall_score: float       # 0-1
    total_records: int
    valid_records: int
    error_count: int
    warning_count: int
    anomaly_count: int
    
    @property
    def quality_grade(self) -> DataQuality:
        """Convert overall score to DataQuality enum."""
        if self.overall_score >= 0.95:
            return DataQuality.HIGH
        elif self.overall_score >= 0.8:
            return DataQuality.MEDIUM
        elif self.overall_score >= 0.6:
            return DataQuality.LOW
        else:
            return DataQuality.STALE


class DataQualityValidator:
    """
    Comprehensive data quality validator for option data.
    
    Provides validation rules, anomaly detection, and quality scoring
    for option market data from various sources.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize validator with configuration.
        
        Args:
            config: Validation configuration parameters
        """
        self.config = config or {}
        
        # Price validation thresholds
        self.min_price = self.config.get('min_price', 0.01)
        self.max_price = self.config.get('max_price', 100000)
        self.max_price_change_pct = self.config.get('max_price_change_pct', 0.5)  # 50%
        
        # Volume validation thresholds  
        self.min_volume = self.config.get('min_volume', 0)
        self.max_volume = self.config.get('max_volume', 1000000)
        
        # Implied volatility thresholds
        self.min_iv = self.config.get('min_iv', 0.01)  # 1%
        self.max_iv = self.config.get('max_iv', 5.0)   # 500%
        
        # Greeks validation thresholds
        self.min_delta = self.config.get('min_delta', -1.0)
        self.max_delta = self.config.get('max_delta', 1.0)
        self.max_gamma = self.config.get('max_gamma', 10.0)
        self.max_vega = self.config.get('max_vega', 100.0)
        self.max_theta = self.config.get('max_theta', 10.0)
        
        # Data freshness threshold (hours)
        self.max_data_age_hours = self.config.get('max_data_age_hours', 24)
        
        # Statistical thresholds
        self.outlier_std_threshold = self.config.get('outlier_std_threshold', 3.0)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.validation_errors: List[ValidationError] = []
        
    def validate_option_data(self, options: List[OptionData]) -> QualityMetrics:
        """
        Comprehensive validation of option data list.
        
        Args:
            options: List of option data to validate
            
        Returns:
            QualityMetrics: Validation results and quality scores
        """
        self.validation_errors.clear()
        
        if not options:
            return QualityMetrics(
                completeness_score=0.0,
                accuracy_score=0.0,
                consistency_score=0.0,
                timeliness_score=0.0,
                validity_score=0.0,
                overall_score=0.0,
                total_records=0,
                valid_records=0,
                error_count=0,
                warning_count=0,
                anomaly_count=0
            )
        
        # Convert to DataFrame for statistical analysis
        df = self._options_to_dataframe(options)
        
        # Run validation rules
        self._validate_price_data(df, options)
        self._validate_volume_data(df, options)
        self._validate_iv_data(df, options)
        self._validate_greeks_data(df, options)
        self._validate_missing_data(df, options)
        self._validate_duplicate_data(df, options)
        self._validate_consistency(df, options)
        self._validate_timeliness(options)
        
        # Detect statistical anomalies
        anomalies = self._detect_statistical_anomalies(df, options)
        
        # Calculate quality scores
        return self._calculate_quality_metrics(len(options), anomalies)
    
    def _options_to_dataframe(self, options: List[OptionData]) -> pd.DataFrame:
        """Convert option data list to pandas DataFrame."""
        data = []
        for opt in options:
            data.append({
                'code': opt.code,
                'market_price': opt.market_price,
                'strike_price': opt.strike_price,
                'volume': opt.volume,
                'open_interest': opt.open_interest,
                'implied_volatility': opt.implied_volatility,
                'theoretical_price': opt.theoretical_price,
                'delta': opt.delta,
                'gamma': opt.gamma,
                'theta': opt.theta,
                'vega': opt.vega,
                'days_to_expiry': opt.days_to_expiry,
                'option_type': opt.option_type.value,
                'underlying': opt.underlying
            })
        return pd.DataFrame(data)
    
    def _validate_price_data(self, df: pd.DataFrame, options: List[OptionData]):
        """Validate price data ranges and consistency."""
        for i, (_, row) in enumerate(df.iterrows()):
            option = options[i]
            
            # Market price validation
            if pd.isna(row['market_price']) or row['market_price'] <= 0:
                self._add_error(
                    ValidationRule.PRICE_RANGE,
                    ValidationSeverity.ERROR,
                    f"Invalid market price: {row['market_price']}",
                    'market_price',
                    row['market_price'],
                    option.code
                )
            elif row['market_price'] < self.min_price or row['market_price'] > self.max_price:
                self._add_error(
                    ValidationRule.PRICE_RANGE,
                    ValidationSeverity.WARNING,
                    f"Price outside normal range: {row['market_price']}",
                    'market_price',
                    row['market_price'],
                    option.code
                )
            
            # Strike price validation
            if pd.isna(row['strike_price']) or row['strike_price'] <= 0:
                self._add_error(
                    ValidationRule.PRICE_RANGE,
                    ValidationSeverity.ERROR,
                    f"Invalid strike price: {row['strike_price']}",
                    'strike_price',
                    row['strike_price'],
                    option.code
                )
            
            # Price consistency checks
            if (not pd.isna(row['theoretical_price']) and 
                not pd.isna(row['market_price']) and
                row['theoretical_price'] > 0 and row['market_price'] > 0):
                
                price_diff = abs(row['market_price'] - row['theoretical_price']) / row['theoretical_price']
                if price_diff > self.max_price_change_pct:
                    self._add_error(
                        ValidationRule.PRICE_CONSISTENCY,
                        ValidationSeverity.WARNING,
                        f"Large price deviation: market={row['market_price']:.2f}, theoretical={row['theoretical_price']:.2f}",
                        'price_consistency',
                        price_diff,
                        option.code
                    )
    
    def _validate_volume_data(self, df: pd.DataFrame, options: List[OptionData]):
        """Validate volume data ranges."""
        for i, (_, row) in enumerate(df.iterrows()):
            option = options[i]
            
            if pd.isna(row['volume']) or row['volume'] < 0:
                self._add_error(
                    ValidationRule.VOLUME_RANGE,
                    ValidationSeverity.WARNING,
                    f"Invalid volume: {row['volume']}",
                    'volume',
                    row['volume'],
                    option.code
                )
            elif row['volume'] > self.max_volume:
                self._add_error(
                    ValidationRule.VOLUME_RANGE,
                    ValidationSeverity.INFO,
                    f"Unusually high volume: {row['volume']}",
                    'volume',
                    row['volume'],
                    option.code
                )
    
    def _validate_iv_data(self, df: pd.DataFrame, options: List[OptionData]):
        """Validate implied volatility data."""
        for i, (_, row) in enumerate(df.iterrows()):
            option = options[i]
            
            if not pd.isna(row['implied_volatility']):
                iv = row['implied_volatility']
                if iv <= 0:
                    self._add_error(
                        ValidationRule.IV_RANGE,
                        ValidationSeverity.ERROR,
                        f"Invalid implied volatility: {iv}",
                        'implied_volatility',
                        iv,
                        option.code
                    )
                elif iv < self.min_iv or iv > self.max_iv:
                    self._add_error(
                        ValidationRule.IV_RANGE,
                        ValidationSeverity.WARNING,
                        f"Implied volatility outside normal range: {iv:.1%}",
                        'implied_volatility',
                        iv,
                        option.code
                    )
    
    def _validate_greeks_data(self, df: pd.DataFrame, options: List[OptionData]):
        """Validate Greeks data ranges."""
        for i, (_, row) in enumerate(df.iterrows()):
            option = options[i]
            
            # Delta validation
            if not pd.isna(row['delta']):
                if row['delta'] < self.min_delta or row['delta'] > self.max_delta:
                    self._add_error(
                        ValidationRule.GREEK_CONSISTENCY,
                        ValidationSeverity.WARNING,
                        f"Delta outside expected range: {row['delta']}",
                        'delta',
                        row['delta'],
                        option.code
                    )
            
            # Gamma validation
            if not pd.isna(row['gamma']) and abs(row['gamma']) > self.max_gamma:
                self._add_error(
                    ValidationRule.GREEK_CONSISTENCY,
                    ValidationSeverity.WARNING,
                    f"Gamma outside expected range: {row['gamma']}",
                    'gamma',
                    row['gamma'],
                    option.code
                )
            
            # Vega validation
            if not pd.isna(row['vega']) and abs(row['vega']) > self.max_vega:
                self._add_error(
                    ValidationRule.GREEK_CONSISTENCY,
                    ValidationSeverity.WARNING,
                    f"Vega outside expected range: {row['vega']}",
                    'vega',
                    row['vega'],
                    option.code
                )
            
            # Theta validation
            if not pd.isna(row['theta']) and abs(row['theta']) > self.max_theta:
                self._add_error(
                    ValidationRule.GREEK_CONSISTENCY,
                    ValidationSeverity.WARNING,
                    f"Theta outside expected range: {row['theta']}",
                    'theta',
                    row['theta'],
                    option.code
                )
    
    def _validate_missing_data(self, df: pd.DataFrame, options: List[OptionData]):
        """Validate for missing critical data."""
        required_fields = ['market_price', 'strike_price']
        
        for i, (_, row) in enumerate(df.iterrows()):
            option = options[i]
            
            for field in required_fields:
                if pd.isna(row[field]):
                    self._add_error(
                        ValidationRule.MISSING_DATA,
                        ValidationSeverity.ERROR,
                        f"Missing required field: {field}",
                        field,
                        None,
                        option.code
                    )
    
    def _validate_duplicate_data(self, df: pd.DataFrame, options: List[OptionData]):
        """Check for duplicate option codes."""
        duplicate_codes = df[df.duplicated('code', keep=False)]['code'].tolist()
        
        for code in set(duplicate_codes):
            self._add_error(
                ValidationRule.DUPLICATE_DATA,
                ValidationSeverity.WARNING,
                f"Duplicate option code found: {code}",
                'code',
                code,
                code
            )
    
    def _validate_consistency(self, df: pd.DataFrame, options: List[OptionData]):
        """Validate internal data consistency."""
        # Check put-call parity relationships for same underlying/strike/expiry
        grouped = df.groupby(['underlying', 'strike_price', 'days_to_expiry'])
        
        for name, group in grouped:
            if len(group) >= 2:  # Has both calls and puts
                calls = group[group['option_type'] == 'C']
                puts = group[group['option_type'] == 'P']
                
                if not calls.empty and not puts.empty:
                    # Simplified put-call parity check
                    for _, call_row in calls.iterrows():
                        for _, put_row in puts.iterrows():
                            price_diff = abs(call_row['market_price'] - put_row['market_price'])
                            if price_diff > call_row['strike_price'] * 0.5:  # Rough consistency check
                                self._add_error(
                                    ValidationRule.PRICE_CONSISTENCY,
                                    ValidationSeverity.INFO,
                                    f"Potential put-call parity violation for {name}",
                                    'price_consistency',
                                    price_diff,
                                    f"{call_row['code']}/{put_row['code']}"
                                )
    
    def _validate_timeliness(self, options: List[OptionData]):
        """Validate data timeliness."""
        current_time = datetime.now()
        
        for option in options:
            # Check how close we are to expiry (data should be more recent near expiry)
            if option.days_to_expiry <= 7:  # Within a week of expiry
                max_age_hours = self.max_data_age_hours / 4  # More frequent updates needed
            else:
                max_age_hours = self.max_data_age_hours
            
            # Note: This is a placeholder as we don't have data timestamp in OptionData
            # In production, you'd compare against actual data timestamp
    
    def _detect_statistical_anomalies(self, df: pd.DataFrame, options: List[OptionData]) -> int:
        """Detect statistical anomalies in the data."""
        anomaly_count = 0
        
        # Detect price outliers
        if len(df) > 10:  # Need sufficient data for statistical analysis
            for field in ['market_price', 'volume', 'implied_volatility']:
                if field in df.columns:
                    field_data = df[field].dropna()
                    if len(field_data) > 5:
                        z_scores = np.abs(stats.zscore(field_data))
                        outliers = field_data[z_scores > self.outlier_std_threshold]
                        
                        for idx, value in outliers.items():
                            anomaly_count += 1
                            option = options[idx]
                            self._add_error(
                                ValidationRule.PRICE_CONSISTENCY,
                                ValidationSeverity.INFO,
                                f"Statistical outlier detected in {field}: {value}",
                                field,
                                value,
                                option.code
                            )
        
        return anomaly_count
    
    def _add_error(self, rule: ValidationRule, severity: ValidationSeverity, 
                   message: str, field: str = None, value: Any = None, record_id: str = None):
        """Add a validation error to the list."""
        error = ValidationError(
            rule=rule,
            severity=severity,
            message=message,
            field=field,
            value=value,
            record_id=record_id
        )
        self.validation_errors.append(error)
        
        # Log based on severity
        if severity == ValidationSeverity.CRITICAL:
            self.logger.critical(message)
        elif severity == ValidationSeverity.ERROR:
            self.logger.error(message)
        elif severity == ValidationSeverity.WARNING:
            self.logger.warning(message)
        else:
            self.logger.debug(message)
    
    def _calculate_quality_metrics(self, total_records: int, anomaly_count: int) -> QualityMetrics:
        """Calculate comprehensive quality metrics."""
        if total_records == 0:
            return QualityMetrics(
                completeness_score=0.0,
                accuracy_score=0.0,
                consistency_score=0.0,
                timeliness_score=1.0,  # Assume recent if no data
                validity_score=0.0,
                overall_score=0.0,
                total_records=0,
                valid_records=0,
                error_count=0,
                warning_count=0,
                anomaly_count=0
            )
        
        # Count errors by severity
        error_count = len([e for e in self.validation_errors if e.severity == ValidationSeverity.ERROR])
        warning_count = len([e for e in self.validation_errors if e.severity == ValidationSeverity.WARNING])
        critical_count = len([e for e in self.validation_errors if e.severity == ValidationSeverity.CRITICAL])
        
        # Calculate component scores (0-1)
        completeness_score = max(0, 1.0 - (error_count * 2 + critical_count * 3) / total_records)
        accuracy_score = max(0, 1.0 - (warning_count + error_count * 2) / total_records)
        consistency_score = max(0, 1.0 - anomaly_count / max(total_records, 1))
        timeliness_score = 1.0  # Placeholder - would calculate from actual timestamps
        validity_score = max(0, 1.0 - error_count / total_records)
        
        # Overall score with weights
        overall_score = (
            completeness_score * 0.25 +
            accuracy_score * 0.25 +
            consistency_score * 0.20 +
            timeliness_score * 0.15 +
            validity_score * 0.15
        )
        
        valid_records = total_records - critical_count - error_count
        
        return QualityMetrics(
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            timeliness_score=timeliness_score,
            validity_score=validity_score,
            overall_score=overall_score,
            total_records=total_records,
            valid_records=max(0, valid_records),
            error_count=error_count + critical_count,
            warning_count=warning_count,
            anomaly_count=anomaly_count
        )
    
    def get_validation_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive validation report.
        
        Returns:
            Dict containing validation summary and details
        """
        errors_by_rule = {}
        errors_by_severity = {}
        
        for error in self.validation_errors:
            # Group by rule
            if error.rule not in errors_by_rule:
                errors_by_rule[error.rule] = []
            errors_by_rule[error.rule].append(error)
            
            # Group by severity
            if error.severity not in errors_by_severity:
                errors_by_severity[error.severity] = []
            errors_by_severity[error.severity].append(error)
        
        return {
            'total_errors': len(self.validation_errors),
            'errors_by_rule': {rule.value: len(errors) for rule, errors in errors_by_rule.items()},
            'errors_by_severity': {sev.value: len(errors) for sev, errors in errors_by_severity.items()},
            'validation_errors': [
                {
                    'rule': error.rule.value,
                    'severity': error.severity.value,
                    'message': error.message,
                    'field': error.field,
                    'record_id': error.record_id,
                    'timestamp': error.timestamp.isoformat()
                } for error in self.validation_errors
            ]
        }
    
    def clear_errors(self):
        """Clear all validation errors."""
        self.validation_errors.clear()


class DataQualityMonitor:
    """
    Monitor and track data quality metrics over time.
    
    Provides quality trend analysis and alerting capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize quality monitor.
        
        Args:
            config: Monitor configuration
        """
        self.config = config or {}
        self.history: List[Tuple[datetime, QualityMetrics]] = []
        self.alert_threshold = self.config.get('alert_threshold', 0.7)
        self.history_limit = self.config.get('history_limit', 1000)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def record_quality_metrics(self, metrics: QualityMetrics):
        """
        Record quality metrics for historical tracking.
        
        Args:
            metrics: Quality metrics to record
        """
        timestamp = datetime.now()
        self.history.append((timestamp, metrics))
        
        # Limit history size
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit:]
        
        # Check for quality alerts
        self._check_quality_alerts(metrics)
    
    def _check_quality_alerts(self, metrics: QualityMetrics):
        """Check if quality metrics trigger alerts."""
        if metrics.overall_score < self.alert_threshold:
            self.logger.warning(
                f"Data quality alert: Overall score {metrics.overall_score:.2%} "
                f"below threshold {self.alert_threshold:.2%}"
            )
        
        if metrics.error_count > 0:
            self.logger.warning(
                f"Data validation errors detected: {metrics.error_count} errors, "
                f"{metrics.warning_count} warnings"
            )
    
    def get_quality_trend(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get quality trend analysis.
        
        Args:
            hours_back: Hours to look back for trend analysis
            
        Returns:
            Dict with trend information
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_metrics = [
            (ts, metrics) for ts, metrics in self.history 
            if ts >= cutoff_time
        ]
        
        if not recent_metrics:
            return {'trend': 'no_data', 'current_quality': None}
        
        # Calculate trend
        scores = [metrics.overall_score for _, metrics in recent_metrics]
        
        if len(scores) >= 2:
            # Simple linear trend
            x = range(len(scores))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, scores)
            
            if slope > 0.01:
                trend = 'improving'
            elif slope < -0.01:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'trend': trend,
            'current_quality': recent_metrics[-1][1].quality_grade.value,
            'avg_score': np.mean(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'data_points': len(scores)
        }