import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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

    print(
        f"Using threshold of {threshold_pct}% for price movement classification with days_to_predict={days_to_predict}")

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
                current_data['ClosePrice'],  # Current price
                current_data['rsi_14'],  # RSI
                current_data['macd'],  # MACD
                current_data['macd_signal'],  # MACD Signal
                current_data['upper_band'],  # Bollinger Upper
                current_data['lower_band'],  # Bollinger Lower
                current_data['obv'],  # On-Balance Volume
                current_data['atr_14'],  # Average True Range
                current_data['volatility_20'],  # Volatility
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
    Train an SVM model for stock prediction

    Parameters:
    X (array): Feature matrix
    y (array): Target vector
    days_to_predict (int): Number of days to predict ahead

    Returns:
    tuple: (model, scaler, accuracy, report, confusion_matrix)
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Kernels và tham số dựa vào khoảng thời gian dự đoán
    kernels = ['linear', 'rbf', 'poly']

    # Điều chỉnh C based on prediction horizon
    if days_to_predict <= 2:
        # Khoảng thời gian ngắn cần một mô hình ít regularization hơn
        C_values = [0.5, 1.0, 2.0]
    else:
        # Khoảng thời gian dài cần một mô hình với nhiều regularization hơn
        C_values = [0.1, 1.0, 5.0]

    best_accuracy = 0
    best_model = None
    best_report = None

    for kernel in kernels:
        for C in C_values:
            model = SVC(kernel=kernel, C=C, gamma='scale', random_state=42, probability=True)
            model.fit(X_train_scaled, y_train)

            # Predict on test set
            y_pred = model.predict(X_test_scaled)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            # Save the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_report = report

    # Get confusion matrix
    y_pred = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)

    # Đảm bảo report có thể serializable
    processed_report = {}
    for key, value in best_report.items():
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

    # Chuyển confusion matrix thành list
    cm_list = cm.tolist()

    # Đảm bảo accuracy là float
    accuracy = float(best_accuracy)

    print(f"Trained SVM model for days_to_predict={days_to_predict} with accuracy: {accuracy:.4f}")

    return best_model, scaler, accuracy, processed_report, cm_list


def predict_stock_movement(model, scaler, features):
    """
    Predict stock movement using trained SVM model

    Parameters:
    model: Trained SVM model
    scaler: Feature scaler
    features (array): Features for prediction

    Returns:
    int: Predicted movement class (-1, 0, 1)
    """
    # Scale features
    features_scaled = scaler.transform([features])

    # Predict
    prediction = model.predict(features_scaled)[0]

    # Probabilities
    probabilities = model.predict_proba(features_scaled)

    # Confidence score
    confidence_score = np.max(probabilities, axis=1)

    return prediction, confidence_score

def get_prediction_label(prediction, confidence=None):
    """Convert prediction class to label with confidence level"""
    confidence_str = f" (Độ tin cậy: {confidence[0]:.2f})" if confidence[0] is not None else ""

    if prediction == 1:
        return f"Tăng giá{confidence_str}", "strong_buy"
    elif prediction == 0:
        return f"Đi ngang{confidence_str}", "hold"
    else:
        return f"Giảm giá{confidence_str}", "strong_sell"

def analyze_stocks_with_svm(stock_data, beta_values, days_to_predict=5):
    """
    Analyze stocks with SVM model to predict price movements

    Parameters:
    stock_data (DataFrame): Historical stock data
    beta_values (DataFrame): Beta values (optional)
    days_to_predict (int): Number of days to predict ahead

    Returns:
    dict: Result of SVM analysis including predictions and metrics
    """
    try:
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

        print(f"Completed SVM analysis with {len(predictions)} predictions for days_ahead={days_to_predict}")

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