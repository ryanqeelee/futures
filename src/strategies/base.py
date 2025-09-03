"""
Base classes and interfaces for arbitrage strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
from pydantic import BaseModel, Field

from ..config.models import ArbitrageOpportunity, StrategyType


class OptionType(str, Enum):
    """Option types."""
    CALL = "C"
    PUT = "P"


class ActionType(str, Enum):
    """Trading action types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class RiskLevel(str, Enum):
    """Risk assessment levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM" 
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class OptionData:
    """Option market data container."""
    code: str
    name: str
    underlying: str
    option_type: OptionType
    strike_price: float
    expiry_date: datetime
    market_price: float
    bid_price: float
    ask_price: float
    volume: int
    open_interest: int
    implied_volatility: Optional[float] = None
    theoretical_price: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    
    @property
    def days_to_expiry(self) -> int:
        """Calculate days to expiration."""
        return (self.expiry_date - datetime.now()).days
    
    @property
    def time_to_expiry(self) -> float:
        """Calculate time to expiration in years."""
        return self.days_to_expiry / 365.0
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price from bid/ask."""
        if self.bid_price and self.ask_price:
            return (self.bid_price + self.ask_price) / 2
        return self.market_price
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        if self.bid_price and self.ask_price:
            return self.ask_price - self.bid_price
        return 0.0
    
    @property
    def spread_pct(self) -> float:
        """Calculate bid-ask spread percentage."""
        spread = self.spread
        if spread > 0 and self.mid_price > 0:
            return spread / self.mid_price
        return 0.0


@dataclass
class TradingAction:
    """Trading action specification."""
    instrument: str
    action: ActionType
    quantity: int
    price: Optional[float] = None
    order_type: str = "LIMIT"
    
    @property
    def value(self) -> float:
        """Calculate action value (positive for buy, negative for sell)."""
        price = self.price or 0
        multiplier = 1 if self.action == ActionType.BUY else -1
        return price * self.quantity * multiplier


@dataclass
class RiskMetrics:
    """Risk assessment metrics."""
    max_loss: float
    max_gain: float
    probability_profit: float
    expected_return: float
    risk_level: RiskLevel
    liquidity_risk: float
    time_decay_risk: float
    volatility_risk: float
    
    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk-reward ratio."""
        if self.max_loss > 0:
            return self.max_gain / self.max_loss
        return float('inf')


class StrategyResult(BaseModel):
    """Strategy execution result."""
    model_config = {'extra': 'forbid'}
    
    strategy_name: str = Field(..., description="Name of the strategy")
    opportunities: List[ArbitrageOpportunity] = Field(default_factory=list, description="Found opportunities")
    execution_time: float = Field(..., description="Execution time in seconds")
    data_timestamp: datetime = Field(..., description="Data timestamp used")
    success: bool = Field(True, description="Whether strategy executed successfully")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class StrategyParameters(BaseModel):
    """Base strategy parameters."""
    model_config = {'extra': 'allow'}  # Allow extra fields for strategy-specific params
    
    min_profit_threshold: float = Field(0.01, ge=0, description="Minimum profit threshold")
    max_risk_tolerance: float = Field(0.1, ge=0, le=1.0, description="Maximum risk tolerance")
    min_liquidity_volume: int = Field(100, ge=0, description="Minimum volume for liquidity")
    max_days_to_expiry: int = Field(90, ge=1, description="Maximum days to expiry")
    min_days_to_expiry: int = Field(1, ge=1, description="Minimum days to expiry")


class BaseStrategy(ABC):
    """
    Abstract base class for arbitrage strategies.
    
    All strategy implementations must inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, parameters: Optional[StrategyParameters] = None):
        """
        Initialize strategy with parameters.
        
        Args:
            parameters: Strategy parameters, uses defaults if None
        """
        self.parameters = parameters or StrategyParameters()
        self._name = self.__class__.__name__
        
    @property
    @abstractmethod
    def strategy_type(self) -> StrategyType:
        """Return the strategy type."""
        pass
    
    @property
    def name(self) -> str:
        """Return strategy name."""
        return self._name
    
    @abstractmethod
    def scan_opportunities(self, options_data: List[OptionData]) -> StrategyResult:
        """
        Scan for arbitrage opportunities in the given option data.
        
        Args:
            options_data: List of option market data
            
        Returns:
            StrategyResult: Scan results with found opportunities
        """
        pass
    
    @abstractmethod
    def calculate_profit(self, options: List[OptionData], actions: List[TradingAction]) -> float:
        """
        Calculate expected profit from a set of trading actions.
        
        Args:
            options: Options involved in the strategy
            actions: Trading actions to execute
            
        Returns:
            float: Expected profit amount
        """
        pass
    
    @abstractmethod 
    def assess_risk(self, options: List[OptionData], actions: List[TradingAction]) -> RiskMetrics:
        """
        Assess risk metrics for a set of trading actions.
        
        Args:
            options: Options involved in the strategy  
            actions: Trading actions to execute
            
        Returns:
            RiskMetrics: Risk assessment results
        """
        pass
    
    def validate_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """
        Validate an arbitrage opportunity against strategy parameters.
        
        Args:
            opportunity: Arbitrage opportunity to validate
            
        Returns:
            bool: True if opportunity meets criteria
        """
        # Check profit threshold
        if opportunity.profit_margin < self.parameters.min_profit_threshold:
            return False
            
        # Check risk tolerance
        if opportunity.risk_score > self.parameters.max_risk_tolerance:
            return False
            
        # Check expiry bounds
        if not (self.parameters.min_days_to_expiry <= opportunity.days_to_expiry <= self.parameters.max_days_to_expiry):
            return False
            
        # Check liquidity
        min_volume = min(opportunity.volumes.values()) if opportunity.volumes else 0
        if min_volume < self.parameters.min_liquidity_volume:
            return False
            
        return True
    
    def filter_options(self, options_data: List[OptionData]) -> List[OptionData]:
        """
        Filter options based on strategy parameters.
        
        Args:
            options_data: Raw options data
            
        Returns:
            List[OptionData]: Filtered options data
        """
        filtered = []
        
        for option in options_data:
            # Check expiry bounds
            if not (self.parameters.min_days_to_expiry <= option.days_to_expiry <= self.parameters.max_days_to_expiry):
                continue
                
            # Check liquidity
            if option.volume < self.parameters.min_liquidity_volume:
                continue
                
            # Check for valid pricing
            if option.market_price <= 0:
                continue
                
            filtered.append(option)
            
        return filtered
    
    def calculate_confidence_score(self, opportunity: ArbitrageOpportunity) -> float:
        """
        Calculate confidence score for an opportunity.
        
        Args:
            opportunity: Arbitrage opportunity
            
        Returns:
            float: Confidence score between 0 and 1
        """
        score = 0.0
        
        # Profit margin factor (higher profit = higher confidence)
        profit_factor = min(opportunity.profit_margin * 10, 0.4)  # Cap at 0.4
        score += profit_factor
        
        # Risk factor (lower risk = higher confidence) 
        risk_factor = max(0, 0.3 - opportunity.risk_score)
        score += risk_factor
        
        # Liquidity factor
        if opportunity.volumes:
            min_volume = min(opportunity.volumes.values())
            volume_factor = min(min_volume / 1000, 0.2)  # Cap at 0.2
            score += volume_factor
            
        # Time factor (more time = more confidence, but diminishing returns)
        time_factor = min(opportunity.days_to_expiry / 180, 0.1)  # Cap at 0.1
        score += time_factor
        
        return min(score, 1.0)
    
    def __str__(self) -> str:
        return f"{self.name}({self.strategy_type.value})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(parameters={self.parameters})"


class StrategyRegistry:
    """
    Registry for strategy implementations.
    
    Allows registration and discovery of strategy implementations.
    """
    
    _strategies: Dict[StrategyType, type] = {}
    _instances: Dict[str, BaseStrategy] = {}
    
    @classmethod
    def register(cls, strategy_type: StrategyType):
        """
        Decorator to register a strategy class.
        
        Args:
            strategy_type: Type of strategy being registered
            
        Returns:
            Decorator function
        """
        def decorator(strategy_class: type) -> type:
            if not issubclass(strategy_class, BaseStrategy):
                raise ValueError(f"Strategy {strategy_class} must inherit from BaseStrategy")
            
            cls._strategies[strategy_type] = strategy_class
            return strategy_class
        return decorator
    
    @classmethod
    def create_strategy(cls, strategy_type: StrategyType, parameters: Optional[StrategyParameters] = None) -> BaseStrategy:
        """
        Create a strategy instance.
        
        Args:
            strategy_type: Type of strategy to create
            parameters: Strategy parameters
            
        Returns:
            BaseStrategy: Strategy instance
            
        Raises:
            ValueError: If strategy type not registered
        """
        if strategy_type not in cls._strategies:
            raise ValueError(f"Strategy type {strategy_type} not registered")
            
        strategy_class = cls._strategies[strategy_type]
        return strategy_class(parameters)
    
    @classmethod
    def get_registered_strategies(cls) -> Dict[StrategyType, type]:
        """
        Get all registered strategy types.
        
        Returns:
            Dict mapping strategy types to their classes
        """
        return cls._strategies.copy()
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear the strategy registry (mainly for testing)."""
        cls._strategies.clear()
        cls._instances.clear()