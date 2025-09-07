# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **futures/options arbitrage discovery system** (期权套利发现系统) built in Python. The system focuses on discovering arbitrage opportunities in options markets with acceptable risk for actual returns. It's a quantitative trading tool with both web interface and command-line capabilities.

## Architecture

The system follows a **three-layer architecture**:
- **UI Layer**: Streamlit web interface (`src/ui/`) with real-time monitoring dashboard
- **Business Layer**: Core arbitrage engine (`src/engine/`), strategies (`src/strategies/`), and risk management
- **Data Layer**: Data adapters (`src/adapters/`), caching system (`src/cache/`), and configuration (`src/config/`)

**Key Components:**
- `src/engine/arbitrage_engine.py` - Core arbitrage discovery engine
- `src/engine/risk_manager.py` - Advanced risk management system
- `src/strategies/` - Multiple arbitrage strategy implementations (pricing, parity, volatility)
- `src/core/plugin_manager.py` - Plugin system for extensibility
- `src/ui/streamlit_app.py` - Main Streamlit web interface

## Development Commands

### Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Web interface (primary method)
python run.py --mode web --port 8501
# OR directly:
streamlit run app.py

# Command line interface
python run.py --mode scan
```

### Testing
```bash
# Run all tests
pytest

# Run specific test files
pytest src/engine/tests/test_arbitrage_engine.py
pytest src/engine/tests/test_risk_manager.py
pytest tests/test_integration.py

# Run integration tests
python test_tushare_integration.py
python validate_plugin_system.py
```

### Code Quality
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy src/
```

## Key Design Patterns

### Strategy Pattern
All arbitrage strategies implement a common interface and are managed by `StrategyManager`. New strategies can be added by extending the base strategy class.

### Plugin Architecture
The system uses a plugin manager (`src/core/plugin_manager.py`) for extensibility. Strategies and data sources are pluggable components.

### Adapter Pattern
Data sources are abstracted through adapters (`src/adapters/`) to support multiple data providers (Tushare, Wind, etc.).

### Caching Strategy
Intelligent caching system (`src/core/cache_manager.py`) with Redis and disk cache to optimize API calls and performance.

## Configuration Management

- Environment variables are loaded from `.env` file
- System settings in `src/config/settings.py`
- Strategy parameters in `src/config/strategy_config.py`
- Risk parameters in `src/config/risk_config.py`
- **Required**: `TUSHARE_TOKEN` environment variable for data access

## Data Sources and APIs

Primary data source is **Tushare API** for real-time and historical options data. The system includes fallback mechanisms and retry logic for API reliability.

## Testing Approach

- **Unit tests**: Individual component testing (pytest)
- **Integration tests**: Cross-component functionality
- **Validation scripts**: Real-world data validation
- **Performance tests**: Benchmark testing for optimization
- Test files follow pattern: `test_*.py`

## Performance Considerations

- **Parallel processing**: Multiple strategies run concurrently
- **Intelligent caching**: Avoid redundant API calls
- **Batch processing**: Minimize network requests
- **Memory optimization**: Efficient data structures for large datasets

## Common Development Patterns

### Adding New Strategies
1. Extend base strategy class in `src/strategies/`
2. Register in `StrategyManager`
3. Add configuration parameters
4. Write unit tests

### Data Source Integration
1. Create adapter in `src/adapters/`
2. Implement standard data interface
3. Add to data source factory
4. Configure fallback mechanisms

### UI Components
Streamlit components are modular and located in `src/ui/components/`. Follow existing component patterns for consistency.

## Important Notes

- The system handles real financial data - ensure proper error handling and validation
- All monetary calculations should account for transaction costs
- Risk management is integrated at multiple levels
- The system supports both development and production configurations
- Extensive logging is implemented throughout the system for debugging and monitoring