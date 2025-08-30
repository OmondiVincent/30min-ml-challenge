import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import json
from datetime import datetime, timedelta

# Generate synthetic user activity logs JSON data
def generate_synthetic_data():
    now = datetime.now()
    data = []
    for i in range(1000):
        timestamp = now - timedelta(days=np.random.randint(0, 60))
        event = {
            "user_id": np.random.randint(1, 50),
            "timestamp": timestamp.isoformat(),
            "event_type": np.random.choice(["login", "logout", "click", "view"]),
            "value": np.random.rand()
        }
        data.append(event)
    return data

# Load data (here from generated data, in practice from JSON file)
def load_data():
    data = generate_synthetic_data()
    return pd.DataFrame(data)

# Transform data
def transform(df, varFiltersCg):
    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Filter by timeframe
    cutoff_date = pd.to_datetime(varFiltersCg)
    df = df[df['timestamp'] >= cutoff_date]
    # Aggregate daily active users and sum of 'value'
    df['date'] = df['timestamp'].dt.date
    agg = df.groupby('date').agg(
        daily_active_users=('user_id', 'nunique'),
        total_value=('value', 'sum')
    ).reset_index()
    return agg

# Train model
def train(df):
    # Features: total_value, predict daily_active_users
    X = df[['total_value']]
    y = df['daily_active_users']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"RMSE: {rmse:.3f}")
    return model

# Serialize model
def save_model(model, path='model.joblib'):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

# Load model (for API)
def load_model(path='model.joblib'):
    return joblib.load(path)

if __name__ == "__main__":
    # Filter: last 30 days
    varFiltersCg = (datetime.now() - timedelta(days=30)).isoformat()
    df = load_data()
    agg = transform(df, varFiltersCg)
    model = train(agg)
    save_model(model)
