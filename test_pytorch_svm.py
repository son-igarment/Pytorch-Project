import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from svm_analysis import analyze_stocks_with_svm

# Function to load and preprocess real stock data
def load_stock_data(csv_file, sample_size=None):
    """
    Load and preprocess real stock data from CSV file
    
    Parameters:
    csv_file (str): Path to the stock data CSV file
    sample_size (int): Number of samples to use (for testing)
    
    Returns:
    DataFrame: Preprocessed stock data
    """
    print(f"Loading stock data from {csv_file}...")
    try:
        # Load the stock data
        stock_data = pd.read_csv(csv_file)
        
        # Rename columns if needed
        if 'TradingDate' in stock_data.columns and 'TradeDate' not in stock_data.columns:
            stock_data = stock_data.rename(columns={'TradingDate': 'TradeDate'})
        
        # Make sure all required columns are present
        required_columns = [
            'MarketCode', 'Ticker', 'TradeDate', 'OpenPrice', 
            'HighestPrice', 'LowestPrice', 'ClosePrice', 'TotalVolume'
        ]
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in stock_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Convert date columns to datetime
        stock_data['TradeDate'] = pd.to_datetime(stock_data['TradeDate'])
        
        # Sort by ticker and date
        stock_data = stock_data.sort_values(['MarketCode', 'Ticker', 'TradeDate'])
        
        # Sample data if sample_size is provided
        if sample_size is not None:
            tickers = stock_data['Ticker'].unique()
            sampled_tickers = np.random.choice(tickers, min(len(tickers), 10), replace=False)
            stock_data = stock_data[stock_data['Ticker'].isin(sampled_tickers)]
            
            # Further limit data points if needed
            if len(stock_data) > sample_size:
                stock_data = stock_data.groupby('Ticker').apply(
                    lambda x: x.sample(min(len(x), sample_size // len(sampled_tickers)))
                ).reset_index(drop=True)
        
        print(f"Loaded {len(stock_data)} records for {len(stock_data['Ticker'].unique())} tickers")
        return stock_data
    
    except Exception as e:
        print(f"Error loading stock data: {str(e)}")
        return generate_sample_data()  # Fallback to generated data

# Generate beta values for real tickers
def generate_beta_values_for_tickers(tickers, market_codes):
    """
    Generate beta values for real tickers
    """
    data = []
    for i, ticker in enumerate(tickers):
        market_code = market_codes[i] if i < len(market_codes) else market_codes[-1]
        beta = np.random.uniform(0.5, 1.5)
        
        # Generate beta interpretation
        if beta < 0.8:
            interpretation = "Độ biến động thấp"
        elif beta < 1.2:
            interpretation = "Độ biến động trung bình"
        else:
            interpretation = "Độ biến động cao"
            
        data.append({
            'stock_code': f'{market_code}:{ticker}',
            'beta': beta,
            'interpretation': interpretation
        })
    
    return pd.DataFrame(data)

# Function to generate sample stock data for testing (fallback if real data issues)
def generate_sample_data(n_samples=500):
    # Generate dates
    dates = pd.date_range(start='2022-01-01', periods=100)
    
    # Real tickers from the CSV (replace with actual tickers from your data)
    tickers = ['VLA', 'MCF', 'BXH', 'X20', 'LHC', 'BCC', 'VNF', 'DDG', 'TTT', 'PBP']
    
    # Create empty dataframe
    data = []
    
    for ticker in tickers:
        # Generate random stock price data
        base_price = np.random.uniform(10000, 50000)
        
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
                'MarketCode': 'HNX',
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

# Plot confusion matrix
def plot_confusion_matrix(cm, class_names=['Giảm', 'Đi ngang', 'Tăng']):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Ma trận nhầm lẫn')
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
    plt.ylabel('Nhãn thực tế')
    plt.xlabel('Nhãn dự đoán')
    plt.savefig('confusion_matrix.png')
    plt.close()

# Plot confidence distribution
def plot_confidence_distribution(predictions):
    confidences = [pred['confidence'] for pred in predictions]
    labels = [int(pred['prediction']) + 1 for pred in predictions]  # Convert -1,0,1 to 0,1,2
    
    plt.figure(figsize=(10, 6))
    
    # Plot confidence distribution for each class
    for class_idx, class_name in enumerate(['Giảm', 'Đi ngang', 'Tăng']):
        class_confidences = [conf for conf, label in zip(confidences, labels) if label == class_idx]
        if class_confidences:
            plt.hist(class_confidences, alpha=0.5, bins=20, label=class_name)
    
    plt.xlabel('Điểm tin cậy')
    plt.ylabel('Tần suất')
    plt.title('Phân phối điểm tin cậy theo nhóm dự đoán')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('confidence_distribution.png')
    plt.close()

if __name__ == "__main__":
    # Check if torch cuda is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Load real stock data from CSV
    stock_data = load_stock_data('StockePrice.csv', sample_size=2000)
    
    # Get unique tickers and market codes
    tickers = stock_data['Ticker'].unique()
    market_codes = stock_data['MarketCode'].unique()
    
    print("Generating beta values for tickers...")
    beta_values = generate_beta_values_for_tickers(tickers, market_codes)
    
    print("Stock data shape:", stock_data.shape)
    print("Beta values shape:", beta_values.shape)
    
    print("\nRunning PyTorch SVM analysis for 5-day prediction horizon...")
    results = analyze_stocks_with_svm(stock_data, beta_values, days_to_predict=5)
    
    if results['success']:
        print("\nPhân tích hoàn tất thành công!")
        print(f"Độ chính xác mô hình: {results['model_metrics']['accuracy']:.4f}")
        print(f"Tổng số dự đoán: {len(results['predictions'])}")
        
        # Display sample predictions with confidence scores
        print("\nMẫu dự đoán:")
        for i, pred in enumerate(results['predictions'][:5]):
            print(f"{i+1}. {pred['stock_code']} ngày {pred['date']}: {pred['prediction_label']} (Tín hiệu: {pred['signal']})")
        
        # Plot confusion matrix
        cm = np.array(results['model_metrics']['confusion_matrix'])
        plot_confusion_matrix(cm)
        print("Ma trận nhầm lẫn đã được lưu dưới dạng 'confusion_matrix.png'")
        
        # Plot confidence distribution
        plot_confidence_distribution(results['predictions'])
        print("Biểu đồ phân phối điểm tin cậy đã được lưu dưới dạng 'confidence_distribution.png'")
        
        # Display class-wise metrics
        print("\nĐộ đo theo lớp:")
        for class_name, class_label in zip(['-1', '0', '1'], ['Giảm', 'Đi ngang', 'Tăng']):
            if class_name in results['model_metrics']['report']:
                metrics = results['model_metrics']['report'][class_name]
                print(f"Lớp {class_label}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1-score={metrics['f1-score']:.4f}")
    else:
        print("Phân tích thất bại:", results['error'])
    
    print("\nKiểm tra các khoảng thời gian dự đoán khác nhau...")
    for days in [2, 10]:
        print(f"\nPhân tích cho khoảng thời gian {days} ngày:")
        results = analyze_stocks_with_svm(stock_data, beta_values, days_to_predict=days)
        
        if results['success']:
            print(f"Độ chính xác mô hình: {results['model_metrics']['accuracy']:.4f}")
            print(f"Tổng số dự đoán: {len(results['predictions'])}")
        else:
            print("Phân tích thất bại:", results['error']) 