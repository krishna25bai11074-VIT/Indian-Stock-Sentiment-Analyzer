# 01_data_collection.ipynb

# ============================================
# STEP 1 — Import Libraries
# ============================================
import yfinance as yf
import pandas as pd
import os

# ============================================
# STEP 2 — Create data folder if not exists
# ============================================
os.makedirs("data", exist_ok=True)

# ============================================
# STEP 3 — Define stocks to download
# ============================================
stocks = {
    "Reliance"  : "RELIANCE.NS",
    "TCS"       : "TCS.NS",
    "Infosys"   : "INFY.NS",
    "HDFC Bank" : "HDFCBANK.NS",
    "Tata Motors": "TATAMOTORS.NS"
}

# ============================================
# STEP 4 — Download and save each stock
# ============================================
for name, symbol in stocks.items():
    print(f"Downloading {name}...")
    data = yf.download(symbol, start="2023-01-01", end="2024-12-31")
    data.to_csv(f"data/{name}.csv", index=True)  # index=True saves Date as column
    print(f"✅ {name} saved — {len(data)} rows\n")

print("🎉 All stocks downloaded successfully!")

# ============================================
# STEP 5 — Preview one stock
# ============================================
reliance = pd.read_csv("data/Reliance.csv")
print(reliance.head(10))
print("\nShape:", reliance.shape)
print("\nColumns:", reliance.columns.tolist())