import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from src.data_loader import load_and_align_usdjpy

# ==================== 設定 ====================
DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

H1_PATH = os.path.join(DATA_DIR, "USDJPY_H1.csv")
H4_PATH = os.path.join(DATA_DIR, "USDJPY_H4.csv")
D1_PATH = os.path.join(DATA_DIR, "USDJPY_D1.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "usdjpy_direction_model.joblib")

print("正在載入並對齊 USDJPY 資料...")
df = load_and_align_usdjpy(H1_PATH, H4_PATH, D1_PATH)
print(f"資料筆數: {len(df)}")

# 特徵與目標
feature_cols = [c for c in df.columns if c not in ['target', 'open', 'high', 'low', 'close', 'volume', 'returns']]
X = df[feature_cols]
y = df['target']

# TimeSeriesSplit 驗證
print("\n開始訓練模型...")
tscv = TimeSeriesSplit(n_splits=5)
accuracies = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Fold {fold+1} 方向準確率: {acc:.4f}")

print(f"\n平均方向準確率: {sum(accuracies)/len(accuracies):.4f}")

# 使用全部資料重新訓練最終模型
final_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
final_model.fit(X, y)

# 儲存模型
joblib.dump(final_model, MODEL_PATH)
print(f"\n模型已儲存至: {MODEL_PATH}")
print("訓練完成！")
