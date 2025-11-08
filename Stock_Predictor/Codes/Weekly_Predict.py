# Weekly Stock Prediction (End of Week Format)
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# ---- STOCK LIST ----
stocks = {
    "NMDC.NS": "NMDC Ltd",
    "NHPC.NS": "NHPC Ltd",
    "IOB.NS": "Indian Overseas Bank",
    "CASTROLIND.NS": "Castrol India",
    "ASHOKLEY.NS": "Ashok Leyland"
}

# ---- INPUT DATES ----
start_input = input("Enter start date (YYYY-MM-DD): ").strip()
end_input = input("Enter end of week date (YYYY-MM-DD): ").strip()

try:
    start_date = datetime.strptime(start_input, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_input, "%Y-%m-%d").date()
except Exception as e:
    raise ValueError("Invalid date format. Use YYYY-MM-DD.") from e

# ---- FUNCTION TO TRAIN + PREDICT ----
def predict_week(symbol, start_date, end_date):
    df = yf.download(symbol, period="6mo", progress=False)
    if df.empty:
        raise RuntimeError(f"No data for {symbol}")

    df = df.reset_index()
    df['Date_only'] = df['Date'].dt.date
    df['Ordinal'] = df['Date_only'].map(datetime.toordinal)

    X_vals = df['Ordinal'].values.reshape(-1, 1)
    y_close = df['Close'].values
    y_high = df['High'].values

    model_close = LinearRegression()
    model_high = LinearRegression()
    model_close.fit(X_vals, y_close)
    model_high.fit(X_vals, y_high)

    # Get last close near start date
    if start_date in df['Date_only'].values:
        last_close = float(df.loc[df['Date_only'] == start_date, 'Close'].values[0])
    else:
        last_close = float(df['Close'].iloc[-1])

    # Generate dates from start_date to end_date
    num_days = (end_date - start_date).days
    future_dates = [start_date + timedelta(days=i) for i in range(1, num_days + 1)]

    preds_close, preds_high = [], []

    for dt in future_dates:
        ord_dt = np.array([[dt.toordinal()]])
        preds_close.append(float(model_close.predict(ord_dt)[0]))
        preds_high.append(float(model_high.predict(ord_dt)[0]))

    # End-of-week predictions
    friday_close = preds_close[-1]
    max_high = max(preds_high)
    max_high_date = future_dates[preds_high.index(max_high)]

    # Calculate change %
    change_pct = ((friday_close - last_close) / last_close) * 100
    trend = "📈 Up" if friday_close > last_close else "📉 Down"

    return last_close, friday_close, max_high, max_high_date, change_pct, trend

# ---- RUN ----
for symbol, pretty in stocks.items():
    try:
        last_close, friday_close, weekly_peak, peak_date, change_pct, trend = predict_week(symbol, start_date, end_date)

        print(f"\n📊 Predicting for: {pretty} ({symbol})")
        print(f"Last Close ({start_date}): ₹{last_close:.2f}")
        print(f"Predicted Close on {end_date}: ₹{friday_close:.2f} {trend} ({change_pct:+.2f}%)")
        print(f"Predicted Weekly Peak: ₹{weekly_peak:.2f} on {peak_date}")
        print("------------------------------------------------------------")

    except Exception as e:
        print(f"\n📊 Predicting for: {pretty} ({symbol}) - ❌ Error: {e}")
        print("------------------------------------------------------------")
