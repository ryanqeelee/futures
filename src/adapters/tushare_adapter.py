"""
Tushare data source adapter implementation.
Integrates with legacy Tushare code while providing new interface.
"""

import os
import asyncio
import logging
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import tushare as ts

from ..config.models import DataSourceType
from ..strategies.base import OptionData, OptionType
from .base import (
    BaseDataAdapter, DataRequest, DataResponse, DataQuality,
    ConnectionStatus, DataSourceError, ConnectionError, 
    AuthenticationError, RateLimitError, DataNotFoundError,
    AdapterRegistry
)
from .data_quality import DataQualityValidator, DataQualityMonitor, QualityMetrics
from ..core.intelligent_cache_manager import TradingCacheManager
from ..core.cache_manager import DataType


def _load_env_file():
    """Load environment variables from .env file (legacy compatibility)."""
    from pathlib import Path
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


def _black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call option pricing (legacy compatibility)."""
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return max(call_price, 0)
    except:
        return 0


def _black_scholes_put(S, K, T, r, sigma):
    """Black-Scholes put option pricing (legacy compatibility)."""
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return max(put_price, 0)
    except:
        return 0


def _implied_volatility(option_price, S, K, T, r, option_type='call', max_iter=100):
    """Calculate implied volatility using Newton's method (legacy compatibility)."""
    try:
        if T <= 0 or S <= 0 or K <= 0 or option_price <= 0:
            return 0
        
        # Initial guess
        sigma = 0.3
        
        for i in range(max_iter):
            if option_type == 'call':
                price = _black_scholes_call(S, K, T, r, sigma)
                # Vega calculation
                d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                vega = S * norm.pdf(d1) * np.sqrt(T)
            else:
                price = _black_scholes_put(S, K, T, r, sigma)
                d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                vega = S * norm.pdf(d1) * np.sqrt(T)
            
            if abs(price - option_price) < 0.0001 or vega == 0:
                break
                
            # Newton's method update
            sigma = sigma - (price - option_price) / vega
            sigma = max(sigma, 0.001)  # Avoid negative volatility
            
        return max(sigma, 0)
    except:
        return 0


@AdapterRegistry.register(DataSourceType.TUSHARE)
class TushareAdapter(BaseDataAdapter):
    """
    Tushare data source adapter.
    
    Integrates with existing Tushare code while providing the new interface.
    Supports both current and historical option data retrieval.
    """
    
    def __init__(self, config: Dict[str, Any], name: Optional[str] = None):
        """
        Initialize Tushare adapter.
        
        Args:
            config: Configuration including api_token
            name: Optional adapter name
        """
        super().__init__(config, name or "TushareAdapter")
        self.api_token = config.get('api_token') or os.getenv('TUSHARE_TOKEN')
        self.risk_free_rate = config.get('risk_free_rate', 0.03)  # Default 3%
        self.max_days_back = config.get('max_days_back', 5)  # Max days to look back for data
        self.request_timeout = config.get('timeout', 30)
        self.max_retries = config.get('retry_count', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        self.rate_limit = config.get('rate_limit', 120)  # requests per minute
        self.batch_size = config.get('batch_size', 100)
        
        # Performance tracking
        self._request_count = 0
        self._last_request_time = 0
        self._request_times = []
        self._cache_requests = 0
        self._cache_hits = 0
        
        # Thread pool for concurrent processing
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Data quality thresholds
        self.min_price_threshold = config.get('min_price_threshold', 0.01)
        self.max_price_threshold = config.get('max_price_threshold', 100000)
        self.min_volume_threshold = config.get('min_volume_threshold', 1)
        self.max_iv_threshold = config.get('max_iv_threshold', 5.0)  # 500% max IV
        
        self._pro_api = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize data quality components
        self.quality_validator = DataQualityValidator(config.get('quality_config', {}))
        self.quality_monitor = DataQualityMonitor(config.get('monitor_config', {}))
        
        # Initialize intelligent cache manager
        cache_config = config.get('cache', {
            'memory': {'max_entries': 5000, 'max_size_mb': 256},
            'disk': {'enabled': True, 'max_size_gb': 1.0},
            'redis': {'enabled': False}
        })
        self.cache_manager = TradingCacheManager(cache_config)
        self._cache_initialized = False
        
    @property
    def data_source_type(self) -> DataSourceType:
        return DataSourceType.TUSHARE
    
    async def connect(self) -> None:
        """
        Establish connection to Tushare.
        
        Raises:
            AuthenticationError: If API token is missing or invalid
            ConnectionError: If connection fails
        """
        try:
            # Load environment file for legacy compatibility
            _load_env_file()
            
            if not self.api_token:
                self.api_token = os.getenv('TUSHARE_TOKEN')
                
            if not self.api_token:
                raise AuthenticationError("TUSHARE_TOKEN not found in config or environment")
            
            # Initialize Tushare
            ts.set_token(self.api_token)
            self._pro_api = ts.pro_api()
            
            # Test connection with a simple query
            test_df = self._pro_api.opt_basic()
            if test_df is None:
                raise ConnectionError("Failed to retrieve test data from Tushare")
            
            # Initialize cache manager
            if not self._cache_initialized:
                await self.cache_manager.initialize()
                self._cache_initialized = True
                self.logger.info("Intelligent cache system initialized")
            
            self._update_connection_status(ConnectionStatus.CONNECTED)
            
        except Exception as e:
            self._update_connection_status(ConnectionStatus.ERROR, str(e))
            if "token" in str(e).lower():
                raise AuthenticationError(f"Tushare authentication failed: {e}")
            else:
                raise ConnectionError(f"Failed to connect to Tushare: {e}")
    
    async def disconnect(self) -> None:
        """Close connection to Tushare."""
        self._pro_api = None
        
        # Shutdown cache manager
        if self._cache_initialized:
            await self.cache_manager.shutdown()
            self._cache_initialized = False
            self.logger.info("Cache system shutdown completed")
        
        self._update_connection_status(ConnectionStatus.DISCONNECTED)
    
    async def get_option_data(self, request: DataRequest) -> DataResponse:
        """
        Retrieve option data from Tushare.
        
        Args:
            request: Data request specification
            
        Returns:
            DataResponse: Retrieved option data
            
        Raises:
            ConnectionError: If not connected
            DataNotFoundError: If no data found
            RateLimitError: If rate limit exceeded
        """
        if not self.is_connected:
            await self.connect()
        
        self.validate_request(request)
        
        # Generate cache key for intelligent cache
        cache_key = self._generate_smart_cache_key(request)
        
        # Determine data type for cache strategy
        data_type = self._determine_data_type(request)
        
        # Try intelligent cache first
        async def data_loader():
            return await self._fetch_option_data_from_source(request)
        
        cached_data = await self.cache_manager.get_with_loader(
            cache_key, data_loader, data_type
        )
        
        if cached_data and isinstance(cached_data, DataResponse):
            self.logger.debug(f"Smart cache hit for request: {cache_key[:16]}...")
            return cached_data
        
        # If we reach here, need to fetch fresh data (cache loader returned None)
        return await self._fetch_option_data_from_source(request)
    
    async def get_underlying_price(self, symbol: str, as_of_date: Optional[date] = None) -> Optional[float]:
        """
        Get underlying asset price from Tushare.
        
        Args:
            symbol: Underlying asset symbol
            as_of_date: Date for historical price
            
        Returns:
            float: Underlying price or None if not found
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            # Try to get futures daily data (most options are on futures)
            trade_date = as_of_date or datetime.now().date()
            
            for days_back in range(self.max_days_back):
                query_date = (trade_date - timedelta(days=days_back)).strftime('%Y%m%d')
                try:
                    # Try futures data first
                    df = self._pro_api.fut_daily(ts_code=symbol, trade_date=query_date)
                    if not df.empty:
                        return float(df['close'].iloc[0])
                except:
                    pass
                
                try:
                    # Try stock data as fallback
                    df = self._pro_api.daily(ts_code=symbol, trade_date=query_date)
                    if not df.empty:
                        return float(df['close'].iloc[0])
                except:
                    pass
            
            return None
            
        except Exception as e:
            print(f"Warning: Failed to get underlying price for {symbol}: {e}")
            return None
    
    async def _get_options_basic(self, request: DataRequest) -> pd.DataFrame:
        """Get basic option information."""
        try:
            # Get all options basic data
            options = self._pro_api.opt_basic()
            if options.empty:
                return pd.DataFrame()
            
            # Filter by instruments if specified
            if request.instruments:
                options = options[options['ts_code'].isin(request.instruments)]
            
            # Filter by underlying assets if specified
            if request.underlying_assets:
                # Extract underlying from option codes
                options['underlying'] = options['ts_code'].str.extract(r'([A-Z]+\d{4})', expand=False)
                options = options[options['underlying'].isin(request.underlying_assets)]
            
            # Filter by option types if specified
            if request.option_types:
                type_values = [t.value for t in request.option_types]
                options = options[options['call_put'].isin(type_values)]
            
            # Calculate days to expiry and filter
            options['delist_date_dt'] = pd.to_datetime(options['delist_date'])
            today = datetime.now()
            options['days_to_expiry'] = (options['delist_date_dt'] - today).dt.days
            
            # Filter by expiry range
            if request.min_days_to_expiry is not None:
                options = options[options['days_to_expiry'] >= request.min_days_to_expiry]
            if request.max_days_to_expiry is not None:
                options = options[options['days_to_expiry'] <= request.max_days_to_expiry]
            
            # Only include options not yet expired
            options = options[options['days_to_expiry'] > 0]
            
            return options
            
        except Exception as e:
            raise DataSourceError(f"Failed to get options basic data: {e}")
    
    async def _get_market_data(self, options_df: pd.DataFrame, as_of_date: Optional[date]) -> pd.DataFrame:
        """Get market data for options."""
        if options_df.empty:
            return pd.DataFrame()
        
        try:
            market_data = None
            
            if as_of_date:
                # Historical data for specific date
                trade_date = as_of_date.strftime('%Y%m%d')
                try:
                    market_data = self._pro_api.opt_daily(trade_date=trade_date)
                except:
                    pass
            else:
                # Try to get recent data
                for days_back in range(self.max_days_back):
                    trade_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
                    try:
                        market_data = self._pro_api.opt_daily(trade_date=trade_date)
                        if not market_data.empty:
                            break
                    except:
                        continue
            
            if market_data is None or market_data.empty:
                return pd.DataFrame()
            
            # Merge with basic data
            full_data = options_df.merge(
                market_data[['ts_code', 'close', 'vol', 'oi', 'trade_date', 'high', 'low', 'open']], 
                on='ts_code', 
                how='inner'
            )
            
            # Filter by volume if requested
            if hasattr(full_data, 'vol'):
                full_data = full_data[full_data['vol'].notna() & (full_data['vol'] > 0)]
                
                # Apply volume filter if specified
                if hasattr(self, 'min_volume') and self.min_volume:
                    full_data = full_data[full_data['vol'] >= self.min_volume]
            
            return full_data
            
        except Exception as e:
            raise DataSourceError(f"Failed to get market data: {e}")
    
    async def _estimate_underlying_prices(self, options_df: pd.DataFrame) -> pd.DataFrame:
        """Estimate underlying prices using put-call parity or direct lookup."""
        if options_df.empty:
            return options_df
        
        options_df = options_df.copy()
        
        # Extract underlying info if not already present
        if 'underlying' not in options_df.columns:
            options_df['underlying'] = options_df['ts_code'].str.extract(r'([A-Z]+\d{4})', expand=False)
        
        estimated_prices = {}
        
        # First try to get actual underlying prices
        for underlying in options_df['underlying'].unique():
            if pd.isna(underlying):
                continue
            
            price = await self.get_underlying_price(underlying)
            if price:
                estimated_prices[underlying] = price
        
        # For underlyings without direct prices, use put-call parity
        for underlying in options_df['underlying'].unique():
            if underlying in estimated_prices or pd.isna(underlying):
                continue
            
            underlying_options = options_df[options_df['underlying'] == underlying]
            
            # Try put-call parity for each strike price
            for strike in underlying_options['exercise_price'].unique():
                if pd.isna(strike):
                    continue
                
                strike_options = underlying_options[underlying_options['exercise_price'] == strike]
                
                calls = strike_options[strike_options['call_put'] == 'C']
                puts = strike_options[strike_options['call_put'] == 'P']
                
                if not calls.empty and not puts.empty:
                    call_price = calls['close'].iloc[0]
                    put_price = puts['close'].iloc[0]
                    
                    # Simple put-call parity: S ≈ C - P + K
                    estimated_S = call_price - put_price + strike
                    if estimated_S > 0:
                        estimated_prices[underlying] = estimated_S
                        break
        
        # Fallback: use strike price as rough estimate
        for underlying in options_df['underlying'].unique():
            if underlying not in estimated_prices and not pd.isna(underlying):
                underlying_options = options_df[options_df['underlying'] == underlying]
                if not underlying_options.empty:
                    # Use the most common strike price as estimate
                    estimated_prices[underlying] = underlying_options['exercise_price'].mode().iloc[0] if not underlying_options['exercise_price'].mode().empty else 1000
        
        # Apply estimated prices
        options_df['underlying_price'] = options_df['underlying'].map(estimated_prices)
        
        return options_df
    
    async def _calculate_pricing_data(self, options_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate theoretical prices and Greeks."""
        if options_df.empty:
            return options_df
        
        options_df = options_df.copy()
        
        theoretical_prices = []
        implied_vols = []
        deltas = []
        gammas = []
        thetas = []
        vegas = []
        
        for _, row in options_df.iterrows():
            try:
                S = row.get('underlying_price')
                K = row.get('exercise_price')
                T = row.get('days_to_expiry', 0) / 365.0
                market_price = row.get('close')
                option_type = 'call' if row.get('call_put') == 'C' else 'put'
                
                if pd.isna(S) or pd.isna(K) or pd.isna(market_price) or S <= 0 or K <= 0 or T <= 0 or market_price <= 0:
                    theoretical_prices.append(np.nan)
                    implied_vols.append(np.nan)
                    deltas.append(np.nan)
                    gammas.append(np.nan)
                    thetas.append(np.nan)
                    vegas.append(np.nan)
                    continue
                
                # Calculate implied volatility
                iv = _implied_volatility(market_price, S, K, T, self.risk_free_rate, option_type)
                implied_vols.append(iv)
                
                # Calculate theoretical price
                if option_type == 'call':
                    theo_price = _black_scholes_call(S, K, T, self.risk_free_rate, iv)
                else:
                    theo_price = _black_scholes_put(S, K, T, self.risk_free_rate, iv)
                theoretical_prices.append(theo_price)
                
                # Calculate Greeks if valid IV
                if iv > 0 and T > 0:
                    d1 = (np.log(S / K) + (self.risk_free_rate + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
                    d2 = d1 - iv * np.sqrt(T)
                    
                    # Delta
                    if option_type == 'call':
                        delta = norm.cdf(d1)
                    else:
                        delta = norm.cdf(d1) - 1
                    deltas.append(delta)
                    
                    # Gamma
                    gamma = norm.pdf(d1) / (S * iv * np.sqrt(T))
                    gammas.append(gamma)
                    
                    # Theta (simplified, per day)
                    theta = -(S * norm.pdf(d1) * iv / (2 * np.sqrt(T)) + 
                            self.risk_free_rate * K * np.exp(-self.risk_free_rate * T) * 
                            (norm.cdf(d2) if option_type == 'call' else norm.cdf(-d2))) / 365
                    thetas.append(theta)
                    
                    # Vega
                    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% volatility change
                    vegas.append(vega)
                else:
                    deltas.append(np.nan)
                    gammas.append(np.nan) 
                    thetas.append(np.nan)
                    vegas.append(np.nan)
                    
            except Exception as e:
                theoretical_prices.append(np.nan)
                implied_vols.append(np.nan)
                deltas.append(np.nan)
                gammas.append(np.nan)
                thetas.append(np.nan)
                vegas.append(np.nan)
        
        options_df['theoretical_price'] = theoretical_prices
        options_df['implied_volatility'] = implied_vols
        options_df['delta'] = deltas
        options_df['gamma'] = gammas
        options_df['theta'] = thetas
        options_df['vega'] = vegas
        
        return options_df
    
    def _convert_to_option_data(self, options_df: pd.DataFrame) -> List[OptionData]:
        """Convert DataFrame to list of OptionData objects with data validation."""
        option_data_list = []
        validation_errors = 0
        
        for _, row in options_df.iterrows():
            try:
                # Validate critical fields
                market_price = float(row.get('close', 0))
                volume = int(row.get('vol', 0))
                
                if not self._validate_price_data(market_price, "market_price"):
                    validation_errors += 1
                    continue
                    
                if not self._validate_volume_data(volume):
                    validation_errors += 1
                    continue
                
                # Validate IV if present
                iv = row.get('implied_volatility')
                if iv is not None and not self._validate_iv_data(iv):
                    validation_errors += 1
                    continue
                
                option_data = OptionData(
                    code=row.get('ts_code', ''),
                    name=row.get('name', ''),
                    underlying=row.get('underlying', ''),
                    option_type=OptionType(row.get('call_put', 'C')),
                    strike_price=float(row.get('exercise_price', 0)),
                    expiry_date=pd.to_datetime(row.get('delist_date')),
                    market_price=market_price,
                    bid_price=0.0,  # Tushare doesn't provide bid/ask, use 0
                    ask_price=0.0,
                    volume=volume,
                    open_interest=int(row.get('oi', 0)),
                    implied_volatility=iv,
                    theoretical_price=row.get('theoretical_price'),
                    delta=row.get('delta'),
                    gamma=row.get('gamma'),
                    theta=row.get('theta'),
                    vega=row.get('vega')
                )
                option_data_list.append(option_data)
            except Exception as e:
                self.logger.warning(f"Failed to convert row to OptionData: {e}")
                validation_errors += 1
                continue
        
        if validation_errors > 0:
            self.logger.warning(f"Data validation failed for {validation_errors} records out of {len(options_df)}")
        
        return option_data_list
    
    def _assess_data_quality(self, options_df: pd.DataFrame) -> DataQuality:
        """Assess the quality of retrieved data."""
        if options_df.empty:
            return DataQuality.LOW
        
        # Calculate completeness scores
        price_completeness = (options_df['close'].notna() & (options_df['close'] > 0)).sum() / len(options_df)
        volume_completeness = (options_df['vol'].notna() & (options_df['vol'] > 0)).sum() / len(options_df)
        
        # Check data freshness (how recent is the data)
        if 'trade_date' in options_df.columns:
            latest_date = pd.to_datetime(options_df['trade_date']).max()
            days_old = (datetime.now() - latest_date).days
            
            if days_old > 7:
                return DataQuality.STALE
        
        # Overall quality assessment
        overall_score = (price_completeness + volume_completeness) / 2
        
        if overall_score >= 0.9:
            return DataQuality.HIGH
        elif overall_score >= 0.7:
            return DataQuality.MEDIUM
        else:
            return DataQuality.LOW
    
    async def get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get real-time market data for multiple symbols.
        
        Args:
            symbols: List of option symbols
            
        Returns:
            Dict mapping symbols to market data
        """
        if not self.is_connected:
            await self.connect()
        
        market_data = {}
        
        try:
            # Get latest option daily data
            for days_back in range(self.max_days_back):
                trade_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
                try:
                    daily_data = await self._retry_request(self._get_daily_data_async, trade_date)
                    if not daily_data.empty:
                        # Filter by requested symbols
                        if symbols:
                            daily_data = daily_data[daily_data['ts_code'].isin(symbols)]
                        
                        for _, row in daily_data.iterrows():
                            symbol = row['ts_code']
                            market_data[symbol] = {
                                'close': float(row['close']),
                                'high': float(row['high']),
                                'low': float(row['low']),
                                'open': float(row['open']),
                                'volume': int(row['vol']),
                                'trade_date': row['trade_date'],
                                'open_interest': int(row.get('oi', 0))
                            }
                        break
                except Exception as e:
                    self.logger.warning(f"Failed to get data for {trade_date}: {e}")
                    continue
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Failed to get market data: {e}")
            raise DataSourceError(f"Failed to get market data: {e}")
    
    async def get_real_time_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get real-time prices for symbols.
        
        Args:
            symbols: List of symbols to get prices for
            
        Returns:
            Dict mapping symbols to current prices
        """
        market_data = await self.get_market_data(symbols)
        return {symbol: data['close'] for symbol, data in market_data.items() if 'close' in data}
    
    async def _get_daily_data_async(self, trade_date: str) -> pd.DataFrame:
        """
        Async wrapper for getting daily data.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._pro_api.opt_daily, trade_date)
    
    async def _batch_request(self, request_func, items: List, batch_size: Optional[int] = None) -> List:
        """
        Execute requests in batches to optimize performance.
        
        Args:
            request_func: Function to execute for each batch
            items: Items to process
            batch_size: Batch size override
            
        Returns:
            List of results
        """
        batch_size = batch_size or self.batch_size
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            try:
                batch_result = await request_func(batch)
                if batch_result:
                    results.extend(batch_result)
            except Exception as e:
                self.logger.warning(f"Batch request failed for items {i}-{i+len(batch)}: {e}")
                # Continue with next batch
                continue
            
            # Rate limiting
            await self._enforce_rate_limit()
        
        return results
    
    async def _enforce_rate_limit(self):
        """
        Enforce API rate limiting.
        """
        current_time = time.time()
        
        # Track request timing
        self._request_count += 1
        self._request_times.append(current_time)
        
        # Remove old requests (beyond 1 minute)
        self._request_times = [t for t in self._request_times if current_time - t < 60]
        
        # Check rate limit
        if len(self._request_times) >= self.rate_limit:
            sleep_time = 60 - (current_time - self._request_times[0])
            if sleep_time > 0:
                self.logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
    
    async def _retry_request(self, request_func, *args, **kwargs):
        """
        Execute request with retry logic.
        
        Args:
            request_func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await request_func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"Request failed after {self.max_retries + 1} attempts: {e}")
        
        raise last_exception
    
    def _validate_price_data(self, price: float, field_name: str = "price") -> bool:
        """
        Validate price data for anomalies.
        
        Args:
            price: Price to validate
            field_name: Name of the field for logging
            
        Returns:
            bool: True if valid
        """
        if pd.isna(price) or price <= 0:
            return False
        
        if price < self.min_price_threshold:
            self.logger.warning(f"Suspicious low {field_name}: {price}")
            return False
        
        if price > self.max_price_threshold:
            self.logger.warning(f"Suspicious high {field_name}: {price}")
            return False
        
        return True
    
    def _validate_volume_data(self, volume: int) -> bool:
        """
        Validate volume data.
        
        Args:
            volume: Volume to validate
            
        Returns:
            bool: True if valid
        """
        if pd.isna(volume) or volume < 0:
            return False
        
        return volume >= self.min_volume_threshold
    
    def _validate_iv_data(self, iv: float) -> bool:
        """
        Validate implied volatility data.
        
        Args:
            iv: Implied volatility to validate
            
        Returns:
            bool: True if valid
        """
        if pd.isna(iv) or iv <= 0:
            return False
        
        if iv > self.max_iv_threshold:
            self.logger.warning(f"Suspicious high IV: {iv}")
            return False
        
        return True
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get adapter performance metrics.
        
        Returns:
            Dict with performance data
        """
        current_time = time.time()
        recent_requests = [t for t in self._request_times if current_time - t < 300]  # Last 5 minutes
        
        avg_response_time = 0
        if len(recent_requests) > 1:
            intervals = [recent_requests[i] - recent_requests[i-1] for i in range(1, len(recent_requests))]
            avg_response_time = sum(intervals) / len(intervals) if intervals else 0
        
        return {
            'total_requests': self._request_count,
            'recent_requests_5min': len(recent_requests),
            'avg_response_time': avg_response_time,
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'connection_status': self.connection_info.status.value,
            'last_error': self.connection_info.last_error
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """
        Calculate cache hit rate.
        
        Returns:
            float: Cache hit rate (0-1)
        """
        if self._cache_requests == 0:
            return 0.0
        
        return self._cache_hits / self._cache_requests
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """
        Get comprehensive data quality report.
        
        Returns:
            Dict with quality metrics and trends
        """
        return {
            'current_metrics': self.quality_validator.get_validation_report(),
            'quality_trend': self.quality_monitor.get_quality_trend(),
            'performance_metrics': self.get_performance_metrics()
        }
    
    async def health_check_comprehensive(self) -> Dict[str, Any]:
        """
        Comprehensive health check including data quality.
        
        Returns:
            Dict with detailed health information
        """
        health_info = {
            'connection_status': self.connection_info.status.value,
            'last_error': self.connection_info.last_error,
            'performance': self.get_performance_metrics(),
            'data_quality': None,
            'api_limits': {
                'rate_limit_status': f"{len(self._request_times)}/{self.rate_limit} requests/min",
                'rate_limit_remaining': max(0, self.rate_limit - len(self._request_times))
            }
        }
        
        try:
            # Test with small data request
            test_request = DataRequest(
                max_days_to_expiry=7,
                min_volume=1
            )
            
            start_time = time.time()
            response = await self.get_option_data(test_request)
            end_time = time.time()
            
            health_info['data_quality'] = {
                'test_successful': True,
                'response_time': end_time - start_time,
                'records_returned': len(response.data),
                'quality_grade': response.quality.value,
                'quality_score': response.metadata.get('quality_metrics', {}).get('overall_score', 0)
            }
            
        except Exception as e:
            health_info['data_quality'] = {
                'test_successful': False,
                'error': str(e)
            }
        
        return health_info
    
    def _generate_smart_cache_key(self, request: DataRequest) -> str:
        """
        Generate intelligent cache key for request.
        
        Args:
            request: Data request
            
        Returns:
            str: Cache key
        """
        # Create comprehensive cache key including all request parameters
        key_components = {
            'instruments': sorted(request.instruments) if request.instruments else None,
            'underlying_assets': sorted(request.underlying_assets) if request.underlying_assets else None,
            'option_types': sorted([t.value for t in request.option_types]) if request.option_types else None,
            'min_days_to_expiry': request.min_days_to_expiry,
            'max_days_to_expiry': request.max_days_to_expiry,
            'min_volume': request.min_volume,
            'include_greeks': request.include_greeks,
            'include_iv': request.include_iv,
            'as_of_date': request.as_of_date.isoformat() if request.as_of_date else None,
            'source': 'tushare'
        }
        
        # Generate hash-based key
        key_str = json.dumps(key_components, sort_keys=True)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()
        
        # Create readable prefix
        prefix_parts = []
        if request.underlying_assets:
            prefix_parts.append('_'.join(request.underlying_assets[:3]))
        if request.option_types:
            prefix_parts.append('_'.join([t.value for t in request.option_types]))
        if request.max_days_to_expiry:
            prefix_parts.append(f"{request.max_days_to_expiry}d")
        
        prefix = '_'.join(prefix_parts) if prefix_parts else 'options'
        
        return f"tushare:{prefix}:{key_hash[:16]}"
    
    def _determine_data_type(self, request: DataRequest) -> DataType:
        """
        Determine cache data type based on request characteristics.
        
        Args:
            request: Data request
            
        Returns:
            DataType: Appropriate data type for caching strategy
        """
        # Real-time data if no specific date requested
        if request.as_of_date is None:
            return DataType.REAL_TIME
        
        # Historical data if specific date requested
        if request.as_of_date < datetime.now().date():
            return DataType.HISTORICAL
        
        # Calculated data if Greeks or IV requested
        if request.include_greeks or request.include_iv:
            return DataType.CALCULATED
        
        # Reference data for basic option information
        return DataType.REFERENCE
    
    async def _fetch_option_data_from_source(self, request: DataRequest) -> DataResponse:
        """
        Fetch option data directly from Tushare (without cache).
        
        Args:
            request: Data request
            
        Returns:
            DataResponse: Fresh data from Tushare
        """
        try:
            # Get options basic data
            options_basic = await self._get_options_basic(request)
            if options_basic.empty:
                raise DataNotFoundError("No options found matching criteria")
            
            # Get market data
            options_with_market = await self._get_market_data(options_basic, request.as_of_date)
            if options_with_market.empty:
                raise DataNotFoundError("No market data found for options")
            
            # Estimate underlying prices
            options_with_underlying = await self._estimate_underlying_prices(options_with_market)
            
            # Calculate theoretical prices and Greeks if requested
            if request.include_iv or request.include_greeks:
                options_with_underlying = await self._calculate_pricing_data(options_with_underlying)
            
            # Convert to OptionData objects
            option_data_list = self._convert_to_option_data(options_with_underlying)
            
            # Comprehensive data quality validation
            quality_metrics = self.quality_validator.validate_option_data(option_data_list)
            self.quality_monitor.record_quality_metrics(quality_metrics)
            
            # Use comprehensive quality assessment
            data_quality = quality_metrics.quality_grade
            
            response = DataResponse(
                request=request,
                data=option_data_list,
                source=self.name,
                quality=data_quality,
                metadata={
                    'total_options': len(option_data_list),
                    'data_date': request.as_of_date or datetime.now().date(),
                    'underlying_count': len(options_with_underlying['underlying'].unique()) if not options_with_underlying.empty else 0,
                    'quality_metrics': {
                        'overall_score': quality_metrics.overall_score,
                        'completeness_score': quality_metrics.completeness_score,
                        'accuracy_score': quality_metrics.accuracy_score,
                        'valid_records': quality_metrics.valid_records,
                        'error_count': quality_metrics.error_count,
                        'warning_count': quality_metrics.warning_count
                    },
                    'validation_errors': self.quality_validator.get_validation_report(),
                    'cache_metadata': {
                        'fetched_at': datetime.now().isoformat(),
                        'source_fresh': True
                    }
                }
            )
            
            return response
            
        except Exception as e:
            if "积分不足" in str(e) or "超出访问次数限制" in str(e):
                raise RateLimitError(f"Tushare rate limit exceeded: {e}", retry_after=3600)
            elif "认证失败" in str(e) or "无效的token" in str(e):
                raise AuthenticationError(f"Tushare authentication error: {e}")
            else:
                raise DataSourceError(f"Failed to retrieve option data: {e}")
    
    async def invalidate_cache(self, pattern: str = "*") -> int:
        """
        Invalidate cache entries matching pattern.
        
        Args:
            pattern: Pattern to match keys (default: all)
            
        Returns:
            int: Number of invalidated entries
        """
        if not self._cache_initialized:
            return 0
        
        return await self.cache_manager.invalidate_pattern(f"tushare:{pattern}")
    
    async def warm_cache_for_underlyings(self, underlyings: List[str]) -> int:
        """
        Warm cache for specific underlying assets.
        
        Args:
            underlyings: List of underlying asset symbols
            
        Returns:
            int: Number of warmed entries
        """
        if not self._cache_initialized:
            return 0
        
        warmed_count = 0
        
        for underlying in underlyings:
            try:
                # Create requests for different scenarios
                requests = [
                    DataRequest(underlying_assets=[underlying], max_days_to_expiry=30),
                    DataRequest(underlying_assets=[underlying], max_days_to_expiry=90, include_iv=True),
                    DataRequest(underlying_assets=[underlying], max_days_to_expiry=7, include_greeks=True)
                ]
                
                for req in requests:
                    cache_key = self._generate_smart_cache_key(req)
                    data_type = self._determine_data_type(req)
                    
                    async def loader():
                        return await self._fetch_option_data_from_source(req)
                    
                    result = await self.cache_manager.get_with_loader(cache_key, loader, data_type)
                    if result:
                        warmed_count += 1
                        
            except Exception as e:
                self.logger.warning(f"Failed to warm cache for {underlying}: {e}")
        
        return warmed_count
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dict with cache performance metrics
        """
        if not self._cache_initialized:
            return {'cache_enabled': False}
        
        stats = self.cache_manager.get_comprehensive_statistics()
        
        # Add adapter-specific metrics
        stats['adapter_metrics'] = {
            'legacy_cache_requests': self._cache_requests,
            'legacy_cache_hits': self._cache_hits,
            'legacy_hit_rate': self._calculate_cache_hit_rate()
        }
        
        return stats
    
    async def cache_health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive cache health check.
        
        Returns:
            Dict with cache health information
        """
        if not self._cache_initialized:
            return {'cache_enabled': False, 'status': 'not_initialized'}
        
        return await self.cache_manager.health_check_comprehensive()