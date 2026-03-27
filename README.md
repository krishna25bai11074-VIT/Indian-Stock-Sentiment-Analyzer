# Indian-Stock-Sentiment-Analyzer

# 📈 Indian Stock Price Predictor

A Machine Learning project that predicts Indian stock prices using historical data from NSE (National Stock Exchange). Built as part of a Data Science course capstone project (BYOP).

---

## 🎯 What Does This Project Do?

This project does **two things**:
- 📊 **Predicts the next day's closing price** (Regression)
- 📈 **Predicts if the stock will go Up or Down** (Classification)

Stocks covered — Reliance, TCS, Infosys, HDFC Bank, Tata Motors

---

## 🛠️ Technologies Used

- Python 3.14
- Pandas — data handling
- NumPy — numerical calculations
- Matplotlib — data visualization
- Scikit-learn — machine learning models
- YFinance — fetching stock data

---

## 📁 Project Structure
```
Indian-Stock-Price-Predictor/
│
├── data/
│   ├── Reliance.csv                 ← raw stock data
│   ├── Reliance_processed.csv       ← cleaned + featured data
│   ├── reliance_price_trend.png     ← price chart
│   ├── reliance_moving_averages.png ← MA chart
│   └── reliance_daily_returns.png   ← returns chart
│
├── Notebook1.py   ← Data Collection
├── Notebook2.py   ← EDA + Feature Engineering
├── Notebook3.py   ← Regression Model
├── Notebook4.py   ← Classification Model
│
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Set Up

**Step 1 — Clone the repository**
```bash
git clone https://github.com/yourusername/Indian-Stock-Price-Predictor.git
cd Indian-Stock-Price-Predictor
```

**Step 2 — Install required libraries**
```bash
pip install pandas numpy matplotlib scikit-learn yfinance
```

**Step 3 — Run notebooks in order**
```bash
python Notebook1.py   # Downloads stock data
python Notebook2.py   # Cleans data + creates features
python Notebook3.py   # Trains regression model
python Notebook4.py   # Trains classification model
```

---

## 📊 Results

### Regression Model (Predicting Exact Price)
| Model | What it does |
|---|---|
| Linear Regression | Predicts next day closing price |
| Random Forest Regressor | Predicts next day closing price |

### Classification Model (Up or Down)
| Model | What it does |
|---|---|
| Logistic Regression | Predicts direction — Up or Down |
| Random Forest Classifier | Predicts direction — Up or Down |

---

## 📉 Sample Charts Generated

- Stock price trend over 2 years
- Moving averages (5, 10, 20 day)
- Daily returns percentage
- Predicted vs Actual price graph
- Confusion matrix for classification

---

## ⚠️ Disclaimer

This project is built for **educational purposes only**.
Stock market predictions are inherently uncertain.
Do NOT use this for actual trading decisions.

---

## 👨‍💻 Author

**Mahes**
B.Tech 1st Year
Connect on GitHub — github.com/yourusername