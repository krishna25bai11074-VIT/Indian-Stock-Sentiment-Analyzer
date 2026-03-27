# 02_eda.ipynb

# ============================================
# STEP 1 — Import Libraries
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================
# STEP 2 — Load data and fix columns
# ============================================
df = pd.read_csv("data/Reliance.csv", header=[0,1])

# Flatten multi-level columns
df.columns = ['_'.join(col).strip() for col in df.columns.values]

# Print columns to verify (you can remove this later)
print("Columns:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))

# Rename columns cleanly
df.rename(columns={
    df.columns[0]        : 'Date',
    'Close_RELIANCE.NS'  : 'Close',
    'Open_RELIANCE.NS'   : 'Open',
    'High_RELIANCE.NS'   : 'High',
    'Low_RELIANCE.NS'    : 'Low',
    'Volume_RELIANCE.NS' : 'Volume'
}, inplace=True)

# Keep only needed columns
df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Drop the first row (it sometimes contains garbage like "Ticker" text)
df = df.iloc[1:].reset_index(drop=True)

# Drop missing values
df = df.dropna()

# Convert types
df['Close']  = pd.to_numeric(df['Close'],  errors='coerce')
df['Open']   = pd.to_numeric(df['Open'],   errors='coerce')
df['High']   = pd.to_numeric(df['High'],   errors='coerce')
df['Low']    = pd.to_numeric(df['Low'],    errors='coerce')
df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

# Fix Date — handles any weird format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop rows where date couldn't be parsed
df = df.dropna(subset=['Date'])

df = df.sort_values('Date').reset_index(drop=True)

print("\n✅ Data loaded successfully!")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# ============================================
# STEP 3 — Plot stock price trend
# ============================================
plt.figure(figsize=(14, 5))
plt.plot(df['Date'], df['Close'], color='blue', linewidth=1.5)
plt.title('Reliance Stock Price (2023-2024)')
plt.xlabel('Date')
plt.ylabel('Closing Price (₹)')
plt.grid(True)
plt.tight_layout()
plt.savefig('data/reliance_price_trend.png')
plt.show()
print("✅ Price trend chart saved!")

# ============================================
# STEP 4 — Add Moving Averages
# ============================================
df['MA5']  = df['Close'].rolling(window=5).mean()
df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA20'] = df['Close'].rolling(window=20).mean()

# Plot with moving averages
plt.figure(figsize=(14, 5))
plt.plot(df['Date'], df['Close'], label='Close Price', linewidth=1)
plt.plot(df['Date'], df['MA5'],   label='5-Day MA',    linewidth=1.5)
plt.plot(df['Date'], df['MA10'],  label='10-Day MA',   linewidth=1.5)
plt.plot(df['Date'], df['MA20'],  label='20-Day MA',   linewidth=1.5)
plt.title('Reliance — Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (₹)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('data/reliance_moving_averages.png')
plt.show()
print("✅ Moving averages chart saved!")

# ============================================
# STEP 5 — Add More Features
# ============================================
df['Daily_Change'] = df['Close'] - df['Open']
df['Daily_Return'] = df['Close'].pct_change() * 100
df['Day_Range']    = df['High'] - df['Low']
df['Day_of_Week']  = df['Date'].dt.dayofweek

# ============================================
# STEP 6 — Plot Daily Returns
# ============================================
plt.figure(figsize=(14, 4))
plt.plot(df['Date'], df['Daily_Return'], color='green', linewidth=1)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Reliance — Daily Returns (%)')
plt.xlabel('Date')
plt.ylabel('Return (%)')
plt.grid(True)
plt.tight_layout()
plt.savefig('data/reliance_daily_returns.png')
plt.show()
print("✅ Daily returns chart saved!")

# ============================================
# STEP 7 — Add Target Columns
# ============================================
df['Target_Price']     = df['Close'].shift(-1)
df['Target_Direction'] = (df['Target_Price'] > df['Close']).astype(int)

# Drop last row (NaN target)
df = df.dropna()

print("\n✅ Features added!")
print(df[['Date','Close','MA5','MA10',
          'Daily_Return','Target_Price',
          'Target_Direction']].tail(5))

# ============================================
# STEP 8 — Save processed data
# ============================================
df.to_csv("data/Reliance_processed.csv", index=False)
print("\n🎉 Processed data saved to data/Reliance_processed.csv")