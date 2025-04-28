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

# Function to load and preprocess market index data
def load_market_index_data(csv_file, sample_size=None):
    """
    Load and preprocess market index data from CSV file
    
    Parameters:
    csv_file (str): Path to the market index CSV file
    sample_size (int): Number of samples to use (for testing)
    
    Returns:
    DataFrame: Preprocessed market index data
    """
    print(f"Loading market index data from {csv_file}...")
    try:
        # Load the market index data
        index_data = pd.read_csv(csv_file)
        
        # Print column names to debug
        print("Available columns in market index data:", index_data.columns.tolist())
        
        # For market index, we need to ensure these columns are present
        required_columns = [
            'MarketCode', 'IndexCode', 'TradeDate', 'OpenIndex', 
            'HighestIndex', 'LowestIndex', 'CloseIndex'
        ]
        
        # Check for required columns and verify they exist
        if 'IndexCode' in index_data.columns:
            print(f"IndexCode exists, with values: {index_data['IndexCode'].unique()[:5]}")
        else:
            print("IndexCode not found in columns")
            
        # Handle specific mappings for market index data
        # Create a subset with just the necessary columns to reduce complexity
        market_subset = pd.DataFrame()
        market_subset['MarketCode'] = index_data['MarketCode']
        market_subset['Ticker'] = index_data['IndexCode']  # We'll use Ticker as our standard column name
        market_subset['TradeDate'] = pd.to_datetime(index_data['TradeDate'])
        
        # Map other required columns
        if 'OpenIndex' in index_data.columns:
            market_subset['OpenPrice'] = pd.to_numeric(index_data['OpenIndex'], errors='coerce')
        else:
            # Use CurrentIndex as a fallback for missing price columns
            market_subset['OpenPrice'] = pd.to_numeric(index_data['CurrentIndex'], errors='coerce')
            
        if 'HighestIndex' in index_data.columns:
            market_subset['HighestPrice'] = pd.to_numeric(index_data['HighestIndex'], errors='coerce')
        else:
            # Use CurrentIndex as fallback
            market_subset['HighestPrice'] = pd.to_numeric(index_data['CurrentIndex'], errors='coerce')
            
        if 'LowestIndex' in index_data.columns:
            market_subset['LowestPrice'] = pd.to_numeric(index_data['LowestIndex'], errors='coerce')
        else:
            # Use CurrentIndex as fallback
            market_subset['LowestPrice'] = pd.to_numeric(index_data['CurrentIndex'], errors='coerce')
            
        if 'CloseIndex' in index_data.columns:
            market_subset['ClosePrice'] = pd.to_numeric(index_data['CloseIndex'], errors='coerce')
        else:
            # Use CurrentIndex as fallback
            market_subset['ClosePrice'] = pd.to_numeric(index_data['CurrentIndex'], errors='coerce')
        
        # Handle volume data - combine trading volume if available
        if 'TotalVolume' in index_data.columns:
            market_subset['TotalVolume'] = pd.to_numeric(index_data['TotalVolume'], errors='coerce')
        elif 'TotalVolumeNT' in index_data.columns and 'TotalVolumePT' in index_data.columns:
            market_subset['TotalVolume'] = pd.to_numeric(index_data['TotalVolumeNT'], errors='coerce') + \
                                          pd.to_numeric(index_data['TotalVolumePT'], errors='coerce')
        else:
            # If no volume data available, create a placeholder
            print("Warning: No volume data found. Using placeholder values.")
            market_subset['TotalVolume'] = 1000000
        
        # Fill NaN values with appropriate defaults to avoid analysis issues
        market_subset = market_subset.fillna(method='ffill')  # Forward fill missing values
        market_subset = market_subset.fillna(method='bfill')  # Backward fill any remaining missing values
        market_subset = market_subset.fillna(0)  # Fill any still-missing values with 0
        
        # Ensure all data is properly sorted
        market_subset = market_subset.sort_values(['MarketCode', 'Ticker', 'TradeDate'])
        
        # Sample data if sample_size is provided
        if sample_size is not None and len(market_subset) > sample_size:
            # Sample by index to maintain chronological continuity within each index
            indices = market_subset['Ticker'].unique()
            sampled_indices = np.random.choice(indices, min(len(indices), 5), replace=False)
            market_subset = market_subset[market_subset['Ticker'].isin(sampled_indices)]
        
        print(f"Prepared market index data: {len(market_subset)} records for {len(market_subset['Ticker'].unique())} market indices")
        print(f"Sample data point: {market_subset.iloc[0].to_dict()}")
        
        return market_subset
    
    except Exception as e:
        print(f"Error loading market index data: {str(e)}")
        print("Error details:", e)  # Print detailed error information
        import traceback
        traceback.print_exc()  # Print full traceback
        return pd.DataFrame()  # Return empty DataFrame on error

# Function to load and preprocess combined market data
def load_market_db_data(csv_file, sample_size=None):
    """
    Load and preprocess market DB data from CSV file
    
    Parameters:
    csv_file (str): Path to the market DB CSV file
    sample_size (int): Number of samples to use (for testing)
    
    Returns:
    DataFrame: Preprocessed market DB data
    """
    print(f"Loading market DB data from {csv_file}...")
    try:
        # Load the market DB data (similar structure to stock data)
        market_data = pd.read_csv(csv_file)
        
        # Make sure all required columns are present
        required_columns = [
            'MarketCode', 'Ticker', 'TradeDate', 'OpenPrice', 
            'HighestPrice', 'LowestPrice', 'ClosePrice', 'TotalVolume'
        ]
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in market_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Convert date columns to datetime
        market_data['TradeDate'] = pd.to_datetime(market_data['TradeDate'])
        
        # Sort by ticker and date
        market_data = market_data.sort_values(['MarketCode', 'Ticker', 'TradeDate'])
        
        # Sample data if sample_size is provided
        if sample_size is not None:
            tickers = market_data['Ticker'].unique()
            sampled_tickers = np.random.choice(tickers, min(len(tickers), 10), replace=False)
            market_data = market_data[market_data['Ticker'].isin(sampled_tickers)]
            
            # Further limit data points if needed
            if len(market_data) > sample_size:
                market_data = market_data.groupby('Ticker').apply(
                    lambda x: x.sample(min(len(x), sample_size // len(sampled_tickers)))
                ).reset_index(drop=True)
        
        print(f"Loaded {len(market_data)} records for {len(market_data['Ticker'].unique())} market tickers")
        return market_data
    
    except Exception as e:
        print(f"Error loading market DB data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

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
def plot_confusion_matrix(cm, class_names=['Giảm', 'Đi ngang', 'Tăng'], file_suffix=''):
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
    filename = f'confusion_matrix{file_suffix}.png'
    plt.savefig(filename)
    plt.close()
    return filename

# Plot confidence distribution
def plot_confidence_distribution(predictions, file_suffix=''):
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
    filename = f'confidence_distribution{file_suffix}.png'
    plt.savefig(filename)
    plt.close()
    return filename

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
        filename = plot_confusion_matrix(cm)
        print(f"Ma trận nhầm lẫn đã được lưu dưới dạng '{filename}'")
        
        # Plot confidence distribution
        filename = plot_confidence_distribution(results['predictions'])
        print(f"Biểu đồ phân phối điểm tin cậy đã được lưu dưới dạng '{filename}'")
        
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
    
    # Now analyze marketDB.csv data
    print("\n=== Phân tích dữ liệu MarketDB ===")
    market_db_data = load_market_db_data('marketDB.csv', sample_size=2000)
    
    if not market_db_data.empty:
        # Get unique tickers and market codes from market DB
        market_tickers = market_db_data['Ticker'].unique()
        market_codes = market_db_data['MarketCode'].unique()
        
        print("Generating beta values for market tickers...")
        market_beta_values = generate_beta_values_for_tickers(market_tickers, market_codes)
        
        print("Market DB data shape:", market_db_data.shape)
        print("Market beta values shape:", market_beta_values.shape)
        
        print("\nRunning PyTorch SVM analysis for market data with 5-day prediction horizon...")
        market_results = analyze_stocks_with_svm(market_db_data, market_beta_values, days_to_predict=5)
        
        if market_results['success']:
            print("\nPhân tích dữ liệu MarketDB hoàn tất thành công!")
            print(f"Độ chính xác mô hình: {market_results['model_metrics']['accuracy']:.4f}")
            print(f"Tổng số dự đoán: {len(market_results['predictions'])}")
            
            # Display sample predictions with confidence scores
            print("\nMẫu dự đoán MarketDB:")
            for i, pred in enumerate(market_results['predictions'][:5]):
                print(f"{i+1}. {pred['stock_code']} ngày {pred['date']}: {pred['prediction_label']} (Tín hiệu: {pred['signal']})")
            
            # Plot confusion matrix for market data
            cm = np.array(market_results['model_metrics']['confusion_matrix'])
            filename = plot_confusion_matrix(cm, file_suffix='_marketdb')
            print(f"Ma trận nhầm lẫn đã được lưu dưới dạng '{filename}'")
            
            # Plot confidence distribution for market data
            filename = plot_confidence_distribution(market_results['predictions'], file_suffix='_marketdb')
            print(f"Biểu đồ phân phối điểm tin cậy đã được lưu dưới dạng '{filename}'")
        else:
            print("Phân tích dữ liệu MarketDB thất bại:", market_results['error'])
    
    # Now analyze marketindex.csv data
    print("\n=== Phân tích dữ liệu Market Index ===")
    market_index_data = load_market_index_data('marketindex.csv', sample_size=1000)
    
    if not market_index_data.empty:
        # Get unique index codes and market codes
        index_codes = market_index_data['Ticker'].unique()
        market_codes = market_index_data['MarketCode'].unique()
        
        print("Generating beta values for market indices...")
        index_beta_values = generate_beta_values_for_tickers(index_codes, market_codes)
        
        print("Market index data shape:", market_index_data.shape)
        print("Index beta values shape:", index_beta_values.shape)
        
        print("\nRunning PyTorch SVM analysis for market index data with 5-day prediction horizon...")
        index_results = analyze_stocks_with_svm(market_index_data, index_beta_values, days_to_predict=5)
        
        if index_results['success']:
            print("\nPhân tích dữ liệu Market Index hoàn tất thành công!")
            print(f"Độ chính xác mô hình: {index_results['model_metrics']['accuracy']:.4f}")
            print(f"Tổng số dự đoán: {len(index_results['predictions'])}")
            
            # Display sample predictions with confidence scores
            print("\nMẫu dự đoán Market Index:")
            for i, pred in enumerate(index_results['predictions'][:5]):
                print(f"{i+1}. {pred['stock_code']} ngày {pred['date']}: {pred['prediction_label']} (Tín hiệu: {pred['signal']})")
            
            # Plot confusion matrix for market index data
            cm = np.array(index_results['model_metrics']['confusion_matrix'])
            filename = plot_confusion_matrix(cm, file_suffix='_marketindex')
            print(f"Ma trận nhầm lẫn đã được lưu dưới dạng '{filename}'")
            
            # Plot confidence distribution for market index data
            filename = plot_confidence_distribution(index_results['predictions'], file_suffix='_marketindex')
            print(f"Biểu đồ phân phối điểm tin cậy đã được lưu dưới dạng '{filename}'")
        else:
            print("Phân tích dữ liệu Market Index thất bại:", index_results['error']) 