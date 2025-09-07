# 预测策略量化指标框架

## 1. 策略性能指标

### 1.1 收益率指标
```python
class StrategyMetrics:
    """策略性能指标计算"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.03) -> float:
        """夏普比率 - 风险调整后收益"""
        excess_returns = returns - risk_free_rate/252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.03) -> float:
        """索提诺比率 - 下行风险调整收益"""
        excess_returns = returns - risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(returns)
        return np.mean(excess_returns) / downside_std * np.sqrt(252)
    
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray) -> float:
        """卡尔马比率 - 年化收益/最大回撤"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        annual_return = (cumulative[-1] ** (252/len(returns))) - 1
        return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    @staticmethod
    def calculate_information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """信息比率 - 超额收益/跟踪误差"""
        excess_returns = returns - benchmark_returns
        tracking_error = np.std(excess_returns)
        return np.mean(excess_returns) / tracking_error if tracking_error > 0 else 0
```

### 1.2 风险指标
```python
class RiskMetrics:
    """风险指标计算"""
    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence: float = 0.05) -> float:
        """风险价值 - 给定置信度下的最大损失"""
        return np.percentile(returns, confidence * 100)
    
    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence: float = 0.05) -> float:
        """条件风险价值 - 超过VaR的平均损失"""
        var = RiskMetrics.calculate_var(returns, confidence)
        return np.mean(returns[returns <= var])
    
    @staticmethod
    def calculate_max_drawdown(returns: np.ndarray) -> Dict[str, float]:
        """最大回撤及相关指标"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        max_dd_idx = np.argmin(drawdown)
        peak_idx = np.argmax(running_max[:max_dd_idx+1])
        
        # 恢复时间计算
        recovery_idx = None
        for i in range(max_dd_idx, len(cumulative)):
            if cumulative[i] >= running_max[max_dd_idx]:
                recovery_idx = i
                break
        
        return {
            'max_drawdown': np.min(drawdown),
            'drawdown_duration': max_dd_idx - peak_idx,
            'recovery_time': recovery_idx - max_dd_idx if recovery_idx else len(returns) - max_dd_idx,
            'current_drawdown': drawdown[-1]
        }
```

### 1.3 预测模型评估指标
```python
class PredictionMetrics:
    """预测模型评估指标"""
    
    @staticmethod
    def directional_accuracy(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """方向性准确率"""
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actuals)
        return np.mean(pred_direction == actual_direction)
    
    @staticmethod
    def hit_ratio(predictions: np.ndarray, actuals: np.ndarray, threshold: float = 0.01) -> float:
        """命中率 - 预测变动幅度超过阈值时的准确率"""
        significant_moves = np.abs(predictions) > threshold
        if np.sum(significant_moves) == 0:
            return 0
        
        correct_predictions = np.sign(predictions[significant_moves]) == np.sign(actuals[significant_moves])
        return np.mean(correct_predictions)
    
    @staticmethod
    def prediction_strength_score(predictions: np.ndarray, actuals: np.ndarray, 
                                confidence_scores: np.ndarray) -> float:
        """预测强度评分 - 考虑置信度的加权准确率"""
        correct = np.sign(predictions) == np.sign(actuals)
        weighted_score = np.average(correct.astype(float), weights=confidence_scores)
        return weighted_score
    
    @staticmethod
    def regression_metrics(predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
        """回归评估指标"""
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r_squared': r_squared
        }
```

## 2. 实时监控指标

### 2.1 策略健康度指标
```python
class HealthMetrics:
    """策略健康度监控"""
    
    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        
    def calculate_strategy_health(self, recent_trades: List[Trade]) -> Dict[str, float]:
        """计算策略健康度"""
        if not recent_trades:
            return {'overall_health': 0.0}
            
        returns = [trade.profit_loss for trade in recent_trades]
        
        # 1. 收益稳定性 (0-1)
        return_stability = 1 - (np.std(returns) / (np.mean(np.abs(returns)) + 1e-8))
        return_stability = max(0, min(1, return_stability))
        
        # 2. 胜率趋势 (0-1)
        win_rate = len([r for r in returns if r > 0]) / len(returns)
        
        # 3. 预测准确率趋势 (0-1)
        prediction_accuracy = self._calculate_recent_accuracy(recent_trades)
        
        # 4. 风险控制效果 (0-1)
        max_single_loss = min(returns) if returns else 0
        expected_max_loss = np.percentile(returns, 5) if len(returns) >= 20 else max_single_loss
        risk_control = 1 - abs(max_single_loss / (abs(expected_max_loss) + 1e-8))
        risk_control = max(0, min(1, risk_control))
        
        # 综合健康度
        overall_health = (return_stability * 0.3 + win_rate * 0.3 + 
                         prediction_accuracy * 0.25 + risk_control * 0.15)
        
        return {
            'overall_health': overall_health,
            'return_stability': return_stability,
            'win_rate': win_rate,
            'prediction_accuracy': prediction_accuracy,
            'risk_control': risk_control
        }

### 2.2 警报系统
class AlertSystem:
    """策略预警系统"""
    
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds
        self.alert_history = []
        
    def check_alerts(self, current_metrics: Dict[str, float]) -> List[Alert]:
        """检查预警条件"""
        alerts = []
        
        # 1. 收益异常预警
        if current_metrics.get('daily_return', 0) < self.thresholds.get('min_daily_return', -0.05):
            alerts.append(Alert('HIGH_LOSS', '单日损失过大', current_metrics['daily_return']))
            
        # 2. 胜率下降预警  
        if current_metrics.get('win_rate', 1) < self.thresholds.get('min_win_rate', 0.5):
            alerts.append(Alert('LOW_WIN_RATE', '胜率持续下降', current_metrics['win_rate']))
            
        # 3. 预测准确率预警
        if current_metrics.get('prediction_accuracy', 1) < self.thresholds.get('min_accuracy', 0.6):
            alerts.append(Alert('LOW_ACCURACY', '预测准确率过低', current_metrics['prediction_accuracy']))
            
        # 4. 回撤预警
        if current_metrics.get('current_drawdown', 0) < self.thresholds.get('max_drawdown', -0.1):
            alerts.append(Alert('HIGH_DRAWDOWN', '回撤超过限制', current_metrics['current_drawdown']))
            
        return alerts

@dataclass
class Alert:
    alert_type: str
    message: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    severity: str = 'MEDIUM'  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
```