
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta


stocks = {
    "NMDC.NS": "NMDC Ltd",
    "NHPC.NS": "NHPC Ltd",
    "IOB.NS": "Indian Overseas Bank",
    "CASTROLIND.NS": "Castrol India",
    "ASHOKLEY.NS": "Ashok Leyland"
}


input_date = input("Enter today's date (YYYY-MM-DD): ").strip()
try:
    today_dt = datetime.strptime(input_date, "%Y-%m-%d").date()
except Exception as e:
    raise ValueError("Invalid date format. Use YYYY-MM-DD.") from e
next_dt = today_dt + timedelta(days=1)


def predict_next_day(symbol, today_date, next_date):
    # fetch 6 months data to have enough points
    df = yf.download(symbol, period="6mo", progress=False)
    if df.empty:
        raise RuntimeError(f"No data for {symbol}")
    df = df.reset_index()
    df['Date_only'] = df['Date'].dt.date

    
    X = df[['Date_only']].copy()
    X['Ordinal'] = X['Date_only'].map(datetime.toordinal)
    X_vals = X['Ordinal'].values.reshape(-1, 1)
    y_close = df['Close'].values
    y_high = df['High'].values

    
    model_close = LinearRegression()
    model_high = LinearRegression()
    model_close.fit(X_vals, y_close)
    model_high.fit(X_vals, y_high)

    
    if today_date in set(df['Date_only'].values):
        last_close_val = float(df.loc[df['Date_only'] == today_date, 'Close'].values[0])
    else:
        last_close_val = float(df['Close'].iloc[-1])  # fallback
        
    
    next_ord = np.array([[next_date.toordinal()]])
    pred_close = float(model_close.predict(next_ord)[0])
    pred_high = float(model_high.predict(next_ord)[0])

    return last_close_val, pred_close, pred_high

import numpy as np


print("\n📈 Next-day Predictions\n")
for symbol, pretty in stocks.items():
    try:
        last_close, predicted_close, predicted_peak = predict_next_day(symbol, today_dt, next_dt)
        diff = predicted_close - last_close
        trend = "📈 Up" if diff > 0 else "📉 Down"
        print(f"{pretty} ({symbol})")
        print(f"Last Close ({today_dt}): ₹{last_close:.2f}")
        print(f"Predicted Next Close ({next_dt}): ₹{predicted_close:.2f} {trend} ({diff:+.2f})")
        print(f"Predicted Next Peak ({next_dt}): ₹{predicted_peak:.2f}")
        print("-" * 45)
    except Exception as e:
        print(f"{pretty} ({symbol}) - ❌ Error: {e}")

