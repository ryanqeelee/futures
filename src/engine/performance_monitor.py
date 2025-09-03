"""
Performance Monitor for Arbitrage Engine.

This module provides real-time performance monitoring, benchmarking,
and optimization tracking for the arbitrage detection system.

Features:
- Real-time performance metrics collection
- Benchmark comparisons (legacy vs enhanced algorithms)
- Resource utilization monitoring
- Performance profiling and bottleneck detection
- Automated optimization suggestions
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import statistics
import logging
import functools

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    name: str
    value: float
    timestamp: datetime
    unit: str = ""
    category: str = "general"


@dataclass 
class BenchmarkResult:
    """Benchmark comparison result."""
    metric_name: str
    legacy_value: float
    enhanced_value: float
    improvement_factor: float
    improvement_percentage: float
    timestamp: datetime


@dataclass
class ResourceUsage:
    """System resource usage snapshot."""
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read: float
    disk_io_write: float
    network_bytes_sent: float
    network_bytes_recv: float
    timestamp: datetime


@dataclass
class ProfilingResult:
    """Function profiling result."""
    function_name: str
    total_time: float
    call_count: int
    avg_time: float
    min_time: float
    max_time: float
    std_time: float


class PerformanceProfiler:
    """Function-level performance profiler."""
    
    def __init__(self):
        self.profiles: Dict[str, List[float]] = defaultdict(list)
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.lock = threading.RLock()
    
    def profile(self, func_name: Optional[str] = None):
        """Decorator to profile function execution time."""
        def decorator(func: Callable) -> Callable:
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    
                    with self.lock:
                        self.profiles[name].append(execution_time)
                        self.call_counts[name] += 1
            
            return wrapper
        return decorator
    
    def get_profile_results(self) -> List[ProfilingResult]:
        """Get profiling results for all functions."""
        results = []
        
        with self.lock:
            for func_name, times in self.profiles.items():
                if times:
                    result = ProfilingResult(
                        function_name=func_name,
                        total_time=sum(times),
                        call_count=len(times),
                        avg_time=statistics.mean(times),
                        min_time=min(times),
                        max_time=max(times),
                        std_time=statistics.stdev(times) if len(times) > 1 else 0
                    )
                    results.append(result)
        
        # Sort by total time descending
        results.sort(key=lambda x: x.total_time, reverse=True)
        return results
    
    def clear_profiles(self):
        """Clear all profiling data."""
        with self.lock:
            self.profiles.clear()
            self.call_counts.clear()


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    Tracks:
    - Execution times and throughput
    - Resource utilization (CPU, memory, I/O)
    - Algorithm performance improvements
    - System bottlenecks and optimization opportunities
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            history_size: Number of historical measurements to keep
        """
        self.history_size = history_size
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics storage
        self.metrics_history: deque = deque(maxlen=history_size)
        self.resource_history: deque = deque(maxlen=history_size)
        self.benchmark_history: deque = deque(maxlen=history_size)
        
        # Current metrics
        self.current_metrics: Dict[str, PerformanceMetric] = {}
        
        # Profiler
        self.profiler = PerformanceProfiler()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_interval = 5.0  # seconds
        
        # Performance targets (based on requirements)
        self.performance_targets = {
            'scan_time': 3.0,  # Target: 3 seconds vs 30 seconds legacy
            'bs_calculation_speedup': 50.0,  # 50x speedup target
            'iv_calculation_speedup': 20.0,  # 20x speedup target
            'memory_usage_mb': 1000.0,  # Max 1GB memory usage
            'cpu_utilization': 80.0,  # Max 80% CPU utilization
        }
        
        # Baseline measurements (for comparison)
        self.baseline_metrics: Dict[str, float] = {}
    
    def start_monitoring(self, interval: float = 5.0):
        """
        Start continuous performance monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring_active:
            return
        
        self.monitoring_interval = interval
        self.monitoring_active = True
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info(f"Performance monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop continuous performance monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10)
        
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                self._collect_resource_metrics()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_resource_metrics(self):
        """Collect system resource metrics."""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read = disk_io.read_bytes if disk_io else 0
            disk_write = disk_io.write_bytes if disk_io else 0
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_sent = network_io.bytes_sent if network_io else 0
            network_recv = network_io.bytes_recv if network_io else 0
            
            resource_usage = ResourceUsage(
                cpu_percent=cpu_percent,
                memory_mb=memory.used / (1024 * 1024),
                memory_percent=memory.percent,
                disk_io_read=disk_read,
                disk_io_write=disk_write,
                network_bytes_sent=network_sent,
                network_bytes_recv=network_recv,
                timestamp=datetime.now()
            )
            
            self.resource_history.append(resource_usage)
            
        except Exception as e:
            self.logger.error(f"Failed to collect resource metrics: {e}")
    
    def record_metric(
        self, 
        name: str, 
        value: float, 
        unit: str = "",
        category: str = "general"
    ):
        """
        Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            category: Metric category
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            unit=unit,
            category=category
        )
        
        self.current_metrics[name] = metric
        self.metrics_history.append(metric)
    
    def measure_execution_time(self, operation_name: str):
        """
        Context manager to measure execution time.
        
        Usage:
            with monitor.measure_execution_time("scan_opportunities"):
                # code to measure
                pass
        """
        return ExecutionTimer(self, operation_name)
    
    def benchmark_against_legacy(
        self, 
        metric_name: str,
        legacy_value: float,
        enhanced_value: float
    ):
        """
        Record benchmark comparison between legacy and enhanced implementations.
        
        Args:
            metric_name: Name of the metric being benchmarked
            legacy_value: Legacy implementation value
            enhanced_value: Enhanced implementation value
        """
        if legacy_value <= 0:
            improvement_factor = float('inf')
            improvement_percentage = 100.0
        else:
            improvement_factor = legacy_value / enhanced_value
            improvement_percentage = ((legacy_value - enhanced_value) / legacy_value) * 100
        
        benchmark = BenchmarkResult(
            metric_name=metric_name,
            legacy_value=legacy_value,
            enhanced_value=enhanced_value,
            improvement_factor=improvement_factor,
            improvement_percentage=improvement_percentage,
            timestamp=datetime.now()
        )
        
        self.benchmark_history.append(benchmark)
        
        self.logger.info(
            f"Benchmark {metric_name}: {improvement_factor:.2f}x speedup "
            f"({improvement_percentage:.1f}% improvement)"
        )
    
    def get_current_metrics(self) -> Dict[str, PerformanceMetric]:
        """Get current performance metrics."""
        return self.current_metrics.copy()
    
    def get_metrics_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get performance metrics summary.
        
        Args:
            time_window: Time window for metrics (None for all)
            
        Returns:
            Dictionary with metrics summary
        """
        now = datetime.now()
        cutoff_time = now - time_window if time_window else datetime.min
        
        # Filter metrics by time window
        filtered_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        # Group metrics by name
        grouped_metrics = defaultdict(list)
        for metric in filtered_metrics:
            grouped_metrics[metric.name].append(metric.value)
        
        # Calculate statistics
        summary = {}
        for name, values in grouped_metrics.items():
            if values:
                summary[name] = {
                    'count': len(values),
                    'latest': values[-1],
                    'average': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                    'trend': self._calculate_trend(values[-10:])  # Last 10 values
                }
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return "stable"
        
        # Simple linear regression to determine trend
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope, _ = np.polyfit(x, y, 1)
        
        if slope > 0.05:  # Arbitrary threshold
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"
    
    def get_resource_utilization(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get system resource utilization summary.
        
        Args:
            time_window: Time window for analysis
            
        Returns:
            Resource utilization summary
        """
        if not self.resource_history:
            return {}
        
        now = datetime.now()
        cutoff_time = now - time_window if time_window else datetime.min
        
        # Filter by time window
        filtered_resources = [
            r for r in self.resource_history 
            if r.timestamp >= cutoff_time
        ]
        
        if not filtered_resources:
            return {}
        
        # Calculate statistics
        cpu_values = [r.cpu_percent for r in filtered_resources]
        memory_values = [r.memory_mb for r in filtered_resources]
        memory_pct_values = [r.memory_percent for r in filtered_resources]
        
        return {
            'cpu': {
                'current': cpu_values[-1],
                'average': statistics.mean(cpu_values),
                'max': max(cpu_values),
                'trend': self._calculate_trend(cpu_values[-10:])
            },
            'memory_mb': {
                'current': memory_values[-1],
                'average': statistics.mean(memory_values),
                'max': max(memory_values),
                'trend': self._calculate_trend(memory_values[-10:])
            },
            'memory_percent': {
                'current': memory_pct_values[-1],
                'average': statistics.mean(memory_pct_values),
                'max': max(memory_pct_values),
                'trend': self._calculate_trend(memory_pct_values[-10:])
            }
        }
    
    def get_benchmark_summary(self) -> Dict[str, BenchmarkResult]:
        """Get latest benchmark results."""
        latest_benchmarks = {}
        
        for benchmark in reversed(self.benchmark_history):
            metric_name = benchmark.metric_name
            if metric_name not in latest_benchmarks:
                latest_benchmarks[metric_name] = benchmark
        
        return latest_benchmarks
    
    def check_performance_targets(self) -> Dict[str, Dict[str, Any]]:
        """
        Check current performance against targets.
        
        Returns:
            Dictionary with target compliance status
        """
        target_status = {}
        current_metrics = self.get_current_metrics()
        resource_util = self.get_resource_utilization(timedelta(minutes=5))
        
        # Check scan time target
        if 'scan_time' in current_metrics:
            scan_time = current_metrics['scan_time'].value
            target = self.performance_targets['scan_time']
            target_status['scan_time'] = {
                'current': scan_time,
                'target': target,
                'met': scan_time <= target,
                'ratio': scan_time / target
            }
        
        # Check memory usage target
        if resource_util and 'memory_mb' in resource_util:
            memory_mb = resource_util['memory_mb']['current']
            target = self.performance_targets['memory_usage_mb']
            target_status['memory_usage'] = {
                'current': memory_mb,
                'target': target,
                'met': memory_mb <= target,
                'ratio': memory_mb / target
            }
        
        # Check CPU utilization target
        if resource_util and 'cpu' in resource_util:
            cpu_pct = resource_util['cpu']['current']
            target = self.performance_targets['cpu_utilization']
            target_status['cpu_utilization'] = {
                'current': cpu_pct,
                'target': target,
                'met': cpu_pct <= target,
                'ratio': cpu_pct / target
            }
        
        return target_status
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': {
                name: {
                    'value': metric.value,
                    'unit': metric.unit,
                    'category': metric.category,
                    'timestamp': metric.timestamp.isoformat()
                }
                for name, metric in self.current_metrics.items()
            },
            'metrics_summary': self.get_metrics_summary(timedelta(hours=1)),
            'resource_utilization': self.get_resource_utilization(timedelta(minutes=15)),
            'benchmark_results': {
                name: {
                    'legacy_value': benchmark.legacy_value,
                    'enhanced_value': benchmark.enhanced_value,
                    'improvement_factor': benchmark.improvement_factor,
                    'improvement_percentage': benchmark.improvement_percentage
                }
                for name, benchmark in self.get_benchmark_summary().items()
            },
            'performance_targets': self.check_performance_targets(),
            'profiling_results': [
                {
                    'function_name': result.function_name,
                    'avg_time': result.avg_time,
                    'total_time': result.total_time,
                    'call_count': result.call_count
                }
                for result in self.profiler.get_profile_results()[:10]  # Top 10
            ]
        }
    
    def suggest_optimizations(self) -> List[str]:
        """Generate optimization suggestions based on performance data."""
        suggestions = []
        
        # Check resource utilization
        resource_util = self.get_resource_utilization(timedelta(minutes=10))
        if resource_util:
            if resource_util['memory_mb']['max'] > 800:  # 800MB threshold
                suggestions.append("Consider implementing data streaming or batch processing to reduce memory usage")
            
            if resource_util['cpu']['average'] > 70:
                suggestions.append("High CPU usage detected - consider optimizing algorithms or adding more parallel processing")
        
        # Check benchmark results
        benchmarks = self.get_benchmark_summary()
        for name, benchmark in benchmarks.items():
            if benchmark.improvement_factor < 2.0:  # Less than 2x improvement
                suggestions.append(f"Low performance improvement in {name} - review optimization strategy")
        
        # Check profiling results
        profiling_results = self.profiler.get_profile_results()
        if profiling_results:
            bottleneck = profiling_results[0]  # Highest time consumer
            if bottleneck.avg_time > 1.0:  # More than 1 second average
                suggestions.append(f"Performance bottleneck detected in {bottleneck.function_name} - consider optimization")
        
        # Check target compliance
        target_status = self.check_performance_targets()
        for target_name, status in target_status.items():
            if not status['met']:
                suggestions.append(f"Performance target not met for {target_name} - current: {status['current']:.2f}, target: {status['target']:.2f}")
        
        return suggestions


class ExecutionTimer:
    """Context manager for measuring execution time."""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            execution_time = time.perf_counter() - self.start_time
            self.monitor.record_metric(
                f"{self.operation_name}_time", 
                execution_time, 
                "seconds",
                "performance"
            )