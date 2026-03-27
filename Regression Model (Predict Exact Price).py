# 03_regression_model.ipynb

# ============================================
# STEP 1 — Import Libraries
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================
# STEP 2 — Load processed data
# ============================================
df = pd.read_csv("data/Reliance_processed.csv")

# ============================================
# STEP 3 — Define Features and Target
# ============================================
features = ['Open', 'High', 'Low', 'Close', 'Volume',
            'MA5', 'MA10', 'MA20',
            'Daily_Change', 'Daily_Return',
            'Day_Range', 'Day_of_Week']

X = df[features]
y = df['Target_Price']

# ============================================
# STEP 4 — Split into Train and Test
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
# shuffle=False is important for time series!

print(f"Training rows : {len(X_train)}")
print(f"Testing rows  : {len(X_test)}")

# ============================================
# STEP 5 — Train Linear Regression
# ============================================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# ============================================
# STEP 6 — Train Random Forest Regressor
# ============================================
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# ============================================
# STEP 7 — Evaluate Both Models
# ============================================
def evaluate_model(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n📊 {name}")
    print(f"   MAE  (Mean Absolute Error) : ₹{mae:.2f}")
    print(f"   RMSE (Root Mean Sq Error)  : ₹{rmse:.2f}")
    print(f"   R²   (Accuracy Score)      : {r2:.4f}")

evaluate_model("Linear Regression",  y_test, lr_predictions)
evaluate_model("Random Forest",       y_test, rf_predictions)

# ============================================
# STEP 8 — Plot Predicted vs Actual
# ============================================
plt.figure(figsize=(14, 5))
plt.plot(y_test.values,    label='Actual Price',          color='blue')
plt.plot(lr_predictions,   label='Linear Regression Pred', color='orange')
plt.plot(rf_predictions,   label='Random Forest Pred',     color='green')
plt.title('Reliance — Actual vs Predicted Price')
plt.xlabel('Days')
plt.ylabel('Price (₹)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('data/regression_results.png')
plt.show()

print("\n✅ Regression model done!")