import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from svm_analysis import analyze_stocks_with_svm

# Function to generate sample stock data for testing
def generate_sample_data(n_samples=500):
    # Generate dates
    dates = pd.date_range(start='2022-01-01', periods=100)
    
    # Generate tickers
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
    
    # Create empty dataframe
    data = []
    
    for ticker in tickers:
        # Generate random stock price data
        base_price = np.random.uniform(50, 500)
        
        for date in dates:
            # Random price movement with trend
            price_change = np.random.normal(0.001, 0.02)
            close_price = base_price * (1 + price_change)
            high_price = close_price * (1 + abs(np.random.normal(0, 0.01)))
            low_price = close_price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = close_price * (1 + np.random.normal(0, 0.01))
            
            # Random volume
            volume = int(np.random.uniform(1000, 1000000))
            
            data.append({
                'MarketCode': 'NYSE',
                'Ticker': ticker,
                'TradeDate': date,
                'OpenPrice': open_price,
                'HighestPrice': high_price,
                'LowestPrice': low_price,
                'ClosePrice': close_price,
                'TotalVolume': volume
            })
            
            # Update base price for next iteration
            base_price = close_price
    
    return pd.DataFrame(data)

# Generate beta values
def generate_beta_values(tickers):
    data = []
    for ticker in tickers:
        beta = np.random.uniform(0.5, 1.5)
        
        # Generate beta interpretation
        if beta < 0.8:
            interpretation = "Low volatility"
        elif beta < 1.2:
            interpretation = "Market-like volatility"
        else:
            interpretation = "High volatility"
            
        data.append({
            'stock_code': f'NYSE:{ticker}',
            'beta': beta,
            'interpretation': interpretation
        })
    
    return pd.DataFrame(data)

# Plot confusion matrix
def plot_confusion_matrix(cm, class_names=['Down', 'Neutral', 'Up']):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations in each cell
    thresh = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

# Plot confidence distribution
def plot_confidence_distribution(predictions):
    confidences = [pred['confidence'] for pred in predictions]
    labels = [int(pred['prediction']) + 1 for pred in predictions]  
    
    plt.figure(figsize=(10, 6))
    
    # Plot confidence distribution for each class
    for class_idx, class_name in enumerate(['Down', 'Neutral', 'Up']):
        class_confidences = [conf for conf, label in zip(confidences, labels) if label == class_idx]
        if class_confidences:
            plt.hist(class_confidences, alpha=0.5, bins=20, label=class_name)
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution by Prediction Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('confidence_distribution.png')
    plt.close()

if __name__ == "__main__":
    # Check if torch cuda is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    print("Generating sample stock data...")
    stock_data = generate_sample_data()
    
    print("Generating sample beta values...")
    tickers = stock_data['Ticker'].unique()
    beta_values = generate_beta_values(tickers)
    
    print("Stock data shape:", stock_data.shape)
    print("Beta values shape:", beta_values.shape)
    
    print("\nRunning PyTorch SVM analysis for 5-day prediction horizon...")
    results = analyze_stocks_with_svm(stock_data, beta_values, days_to_predict=5)
    
    if results['success']:
        print("\nAnalysis completed successfully!")
        print(f"Model accuracy: {results['model_metrics']['accuracy']:.4f}")
        print(f"Total predictions: {len(results['predictions'])}")
        
        # Display sample predictions with confidence scores
        print("\nSample predictions:")
        for i, pred in enumerate(results['predictions'][:5]):
            print(f"{i+1}. {pred['stock_code']} on {pred['date']}: {pred['prediction_label']} (Signal: {pred['signal']})")
        
        # Plot confusion matrix
        cm = np.array(results['model_metrics']['confusion_matrix'])
        plot_confusion_matrix(cm)
        print("Confusion matrix plot saved as 'confusion_matrix.png'")
        
        # Plot confidence distribution
        plot_confidence_distribution(results['predictions'])
        print("Confidence distribution plot saved as 'confidence_distribution.png'")
        
        # Display class-wise metrics
        print("\nClass-wise metrics:")
        for class_name in ['-1', '0', '1']:
            if class_name in results['model_metrics']['report']:
                metrics = results['model_metrics']['report'][class_name]
                print(f"Class {class_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1-score={metrics['f1-score']:.4f}")
    else:
        print("Analysis failed:", results['error'])
    
    print("\nTesting different prediction horizons...")
    for days in [2, 10]:
        print(f"\nRunning analysis for {days}-day prediction horizon...")
        results = analyze_stocks_with_svm(stock_data, beta_values, days_to_predict=days)
        
        if results['success']:
            print(f"Model accuracy: {results['model_metrics']['accuracy']:.4f}")
            print(f"Total predictions: {len(results['predictions'])}")
        else:
            print("Analysis failed:", results['error']) 