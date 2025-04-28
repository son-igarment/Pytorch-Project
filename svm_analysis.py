import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from flask import jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class SVMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SVMModel, self).__init__()
        # More sophisticated model with hidden layers
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 3)  # 3 classes: -1, 0, 1
        )
        
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs.data, 1)
            # Map 0,1,2 to -1,0,1
            return predicted.numpy() - 1

def prepare_features(stock_data, beta_values, days_to_predict=5):
    """
    Prepare features for SVM analysis from stock data and beta values
    
    Parameters:
    stock_data (DataFrame): Historical stock data
    beta_values (DataFrame): Beta values for the stocks
    days_to_predict (int): Number of days to use for prediction horizon
    
    Returns:
    DataFrame: Features and target variables for SVM
    """
    # Create a copy of the data
    df = stock_data.copy()
    
    # Convert date strings to datetime
    df['TradeDate'] = pd.to_datetime(df['TradeDate'])
    
    # Sort by date
    df = df.sort_values(['MarketCode', 'Ticker', 'TradeDate'])
    
    # Group by stock code to process each stock individually
    grouped = df.groupby(['MarketCode', 'Ticker'])
    
    # Initialize lists to store results
    all_features = []
    all_targets = []
    all_stock_codes = []
    all_dates = []
    
    # Điều chỉnh ngưỡng phần trăm dựa trên days_to_predict
    if days_to_predict <= 2:
        threshold_pct = 0.5  # Ngưỡng thấp hơn cho dự đoán ngắn hạn
    else:
        threshold_pct = 1.0  # Ngưỡng mặc định
        
    print(f"Using threshold of {threshold_pct}% for price movement classification with days_to_predict={days_to_predict}")
    
    for code, group in grouped:
        # Skip if less than 10 data points
        if len(group) < 10:
            continue
        
        # Get beta value for this stock
        stock_code = ':'.join(code)
        beta_value = None
        if beta_values is not None:
            beta_row = beta_values[beta_values['stock_code'] == stock_code]
            if not beta_row.empty:
                beta_value = beta_row.iloc[0]['beta']
        
        # Calculate technical indicators
        group = calculate_technical_indicators(group)
        
        # Drop rows with NaN (due to rolling calculations)
        group = group.dropna()
        
        # Prepare features
        for i in range(len(group) - days_to_predict):
            # Current data point
            current_data = group.iloc[i]
            
            # Features
            features = [
                current_data['ClosePrice'],         # Current price
                current_data['rsi_14'],               # RSI
                current_data['macd'],                 # MACD
                current_data['macd_signal'],          # MACD Signal
                current_data['upper_band'],           # Bollinger Upper
                current_data['lower_band'],           # Bollinger Lower
                current_data['obv'],                  # On-Balance Volume
                current_data['atr_14'],               # Average True Range
                current_data['volatility_20'],        # Volatility
            ]
            
            # Add beta as a feature if available
            if beta_value is not None:
                features.append(beta_value)
            
            # Target: Will the price go up in the next 'days_to_predict' days?
            future_price = float(group.iloc[i + days_to_predict]['ClosePrice'])
            current_price = float(current_data['ClosePrice'])
            
            # Classify as 1 (up), 0 (neutral), -1 (down) with adjusted thresholds
            percent_change = (future_price - current_price) / current_price * 100
            
            if percent_change > threshold_pct:  # Up more than threshold_pct
                target = 1
            elif percent_change < -threshold_pct:  # Down more than threshold_pct
                target = -1
            else:  # Between -threshold_pct and threshold_pct
                target = 0
            
            all_features.append(features)
            all_targets.append(target)
            all_stock_codes.append(stock_code)
            all_dates.append(current_data['TradeDate'])
    
    # Convert lists to arrays
    X = np.array(all_features)
    y = np.array(all_targets)
    
    return X, y, all_stock_codes, all_dates

def calculate_technical_indicators(df):
    """Calculate various technical indicators for the dataframe"""
    # Price and volume data
    df = df.copy()
    
    # RSI (Relative Strength Index)
    delta = df['ClosePrice'].astype(float).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    exp1 = df['ClosePrice'].ewm(span=12, adjust=False).mean()
    exp2 = df['ClosePrice'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['20sma'] = df['ClosePrice'].rolling(window=20).mean()
    df['volatility_20'] = df['ClosePrice'].rolling(window=20).std()
    df['upper_band'] = df['20sma'] + (df['volatility_20'] * 2)
    df['lower_band'] = df['20sma'] - (df['volatility_20'] * 2)
    
    # OBV (On-Balance Volume)
    df['daily_ret'] = df['ClosePrice'].astype(float).pct_change()
    df['direction'] = np.where(df['daily_ret'] > 0, 1, -1)
    df.loc[df['daily_ret'] == 0, 'direction'] = 0
    df['obv'] = (pd.to_numeric(df['TotalVolume']) * df['direction']).cumsum()
    
    # ATR (Average True Range)
    df['high_low'] = df['HighestPrice'].astype(float) - df['LowestPrice'].astype(float)
    df['high_close'] = abs(df['HighestPrice'].astype(float) - df['ClosePrice'].astype(float).shift())
    df['low_close'] = abs(df['LowestPrice'].astype(float) - df['ClosePrice'].astype(float).shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr_14'] = df['tr'].rolling(window=14).mean()
    
    return df

def train_svm_model(X, y, days_to_predict=5):
    """
    Train a PyTorch SVM model for stock prediction
    
    Parameters:
    X (array): Feature matrix
    y (array): Target vector
    days_to_predict (int): Number of days to predict ahead
    
    Returns:
    tuple: (model, scaler, accuracy, report, confusion_matrix)
    """
    # Convert target classes (-1, 0, 1) to (0, 1, 2) for PyTorch
    y_adjusted = y + 1
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_adjusted, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    batch_size = min(64, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_dim = X_train_scaled.shape[1]
    model = SVMModel(input_dim)
    
    # Loss function with class weighting to handle imbalance
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Adjust training parameters based on prediction horizon
    if days_to_predict <= 2:
        learning_rate = 0.005
        weight_decay = 0.01
        num_epochs = 150
    else:
        learning_rate = 0.001
        weight_decay = 0.02
        num_epochs = 200
    
    # Use Adam optimizer for better convergence
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training loop with early stopping
    best_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        
        # Update learning rate based on loss
        scheduler.step(epoch_loss)
        
        # Early stopping check
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            # Save the best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model state if early stopping occurred
    if patience_counter >= patience:
        model.load_state_dict(best_model_state)
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        y_pred = predicted.numpy()
    
    # Convert predictions back to original scale (-1, 0, 1)
    y_pred_original = y_pred - 1
    y_test_original = y_test - 1
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_original, y_pred_original)
    report = classification_report(y_test_original, y_pred_original, output_dict=True)
    cm = confusion_matrix(y_test_original, y_pred_original)
    
    # Ensure report is serializable
    processed_report = {}
    for key, value in report.items():
        if isinstance(value, dict):
            processed_report[key] = {}
            for metric, metric_value in value.items():
                if hasattr(metric_value, 'item') or hasattr(metric_value, 'tolist'):
                    processed_report[key][metric] = float(metric_value)
                else:
                    processed_report[key][metric] = metric_value
        else:
            if hasattr(value, 'item') or hasattr(value, 'tolist'):
                processed_report[key] = float(value)
            else:
                processed_report[key] = value
    
    # Convert confusion matrix to list
    cm_list = cm.tolist()
    
    # Convert accuracy to float
    accuracy = float(accuracy)
    
    print(f"Trained PyTorch SVM model for days_to_predict={days_to_predict} with accuracy: {accuracy:.4f}")
    
    return model, scaler, accuracy, processed_report, cm_list

def predict_stock_movement(model, scaler, features):
    """
    Predict stock movement using trained PyTorch SVM model
    
    Parameters:
    model: Trained PyTorch SVM model
    scaler: Feature scaler
    features (array): Features for prediction
    
    Returns:
    int: Predicted movement class (-1, 0, 1)
    float: Confidence score
    """
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Convert to PyTorch tensor
    features_tensor = torch.FloatTensor(features_scaled)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)
        prediction = predicted.item() - 1  # Convert back to -1, 0, 1
        confidence_score = confidence.item()
    
    return prediction, confidence_score

def get_prediction_label(prediction, confidence=None):
    """Convert prediction class to label with confidence level"""
    confidence_str = f" (Độ tin cậy: {confidence:.2f})" if confidence is not None else ""
    
    if prediction == 1:
        return f"Tăng giá{confidence_str}", "strong_buy"
    elif prediction == 0:
        return f"Đi ngang{confidence_str}", "hold"
    else:
        return f"Giảm giá{confidence_str}", "strong_sell"

def analyze_stocks_with_svm(stock_data, beta_values, days_to_predict=5):
    """
    Analyze stocks with PyTorch SVM model to predict price movements

    Parameters:
    stock_data (DataFrame): Historical stock data
    beta_values (DataFrame): Beta values (optional)
    days_to_predict (int): Number of days to predict ahead

    Returns:
    dict: Result of SVM analysis including predictions and metrics
    """
    try:
        # Set device to GPU if available, otherwise CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Prepare features
        X, y, stock_codes, dates = prepare_features(stock_data, beta_values, days_to_predict)

        if len(X) == 0 or len(y) == 0:
            return {
                "success": False,
                "error": "Insufficient data for analysis after filtering"
            }

        # Train SVM model
        model, scaler, accuracy, report, cm = train_svm_model(X, y, days_to_predict)

        # Predict for all samples
        predictions = []
        for i, features in enumerate(X):
            prediction, confidence = predict_stock_movement(model, scaler, features)
            label, signal = get_prediction_label(prediction, confidence)

            # Get beta value and interpretation if available
            beta = None
            beta_interpretation = None
            if beta_values is not None and not beta_values.empty:
                beta_row = beta_values[beta_values['stock_code'] == stock_codes[i]]
                if not beta_row.empty:
                    beta = beta_row.iloc[0]['beta']
                    beta_interpretation = beta_row.iloc[0]['interpretation']

            predictions.append({
                'stock_code': stock_codes[i],
                'date': str(dates[i]),
                'prediction': str(prediction),
                'prediction_label': label,
                'signal': signal,
                'confidence': float(confidence),
                'beta': float(beta) if beta is not None else None,
                'beta_interpretation': beta_interpretation
            })

        # Sort predictions by stock code and date
        predictions = sorted(predictions, key=lambda x: (x['stock_code'], x['date']))

        print(f"Completed PyTorch SVM analysis with {len(predictions)} predictions for days_ahead={days_to_predict}")

        return {
            "success": True,
            "model_metrics": {
                "accuracy": accuracy,
                "report": report,
                "confusion_matrix": cm
            },
            "predictions": predictions,
            "days_ahead": days_to_predict,
            "beta_used": beta_values is not None
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }