# PyTorch SVM for Stock Analysis

## Overview
This project implements a PyTorch-based Support Vector Machine model for stock price movement prediction. The model is designed to analyze stock market data and predict future price movements (up, neutral, or down) based on historical data and technical indicators.

## Features
- Uses PyTorch as the deep learning framework
- Implements SVM (Support Vector Machine) classification using neural networks
- Incorporates technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Predicts stock price movements for multiple time horizons
- Includes confidence scores with predictions
- Visualization tools for model analysis

## Files
- `svm_analysis.py`: Main implementation of the PyTorch SVM model
- `test_pytorch_svm.py`: Test script to demonstrate the model's functionality

## Requirements
- Python 3.6+
- PyTorch
- pandas
- numpy
- matplotlib
- scikit-learn

## Usage

### Basic Analysis
Run the test script to see the model in action with sample data:
```
python test_pytorch_svm.py
```

### Using with Real Data
To use the model with your own stock data:

```python
import pandas as pd
from svm_analysis import analyze_stocks_with_svm

# Load your stock data (requires specific format)
stock_data = pd.read_csv('your_stock_data.csv')

# Optional: Load beta values
beta_values = pd.read_csv('your_beta_values.csv')

# Run analysis
results = analyze_stocks_with_svm(stock_data, beta_values, days_to_predict=5)

# Process results
if results['success']:
    print(f"Model accuracy: {results['model_metrics']['accuracy']}")
    
    # Display predictions
    for pred in results['predictions']:
        print(f"{pred['stock_code']} on {pred['date']}: {pred['prediction_label']}")
```

## Input Data Format
The model expects stock data with the following columns:
- MarketCode: Market identifier
- Ticker: Stock symbol
- TradeDate: Date of trading
- OpenPrice: Opening price
- HighestPrice: Highest price during the day
- LowestPrice: Lowest price during the day
- ClosePrice: Closing price
- TotalVolume: Trading volume

The beta values dataframe should contain:
- stock_code: Combined market code and ticker (e.g., "NYSE:AAPL")
- beta: Beta value of the stock
- interpretation: Interpretation of the beta value

## Model Information
The model uses a neural network architecture with:
- Multiple hidden layers with ReLU activation
- Dropout layers for regularization
- Cross-entropy loss function with class weighting
- Adam optimizer
- Learning rate scheduling and early stopping

## Output
The model provides:
- Prediction classification (-1: down, 0: neutral, 1: up)
- Confidence scores for each prediction
- Model performance metrics including accuracy, precision, recall, and F1-score
- Confusion matrix visualization
- Confidence distribution analysis 