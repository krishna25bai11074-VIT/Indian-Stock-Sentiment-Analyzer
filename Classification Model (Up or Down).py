# 04_classification_model.ipynb

# ============================================
# STEP 1 — Import Libraries
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)

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
y = df['Target_Direction']  # 1=Up, 0=Down

print("Up days   :", y.sum())
print("Down days :", (y==0).sum())

# ============================================
# STEP 4 — Split Data
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ============================================
# STEP 5 — Train Logistic Regression
# ============================================
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# ============================================
# STEP 6 — Train Random Forest Classifier
# ============================================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# ============================================
# STEP 7 — Evaluate Both Models
# ============================================
print("\n📊 Logistic Regression")
print(f"   Accuracy : {accuracy_score(y_test, lr_pred)*100:.2f}%")
print(classification_report(y_test, lr_pred,
      target_names=['Down 📉', 'Up 📈']))

print("\n📊 Random Forest Classifier")
print(f"   Accuracy : {accuracy_score(y_test, rf_pred)*100:.2f}%")
print(classification_report(y_test, rf_pred,
      target_names=['Down 📉', 'Up 📈']))

# ============================================
# STEP 8 — Plot Confusion Matrix
# ============================================
def plot_confusion(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Down','Up'])
    ax.set_yticklabels(['Down','Up'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {name}')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j],
                    ha='center', va='center',
                    color='black', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'data/confusion_{name}.png')
    plt.show()

plot_confusion("Logistic_Regression", y_test, lr_pred)
plot_confusion("Random_Forest",        y_test, rf_pred)

print("\n✅ Classification model done!")
