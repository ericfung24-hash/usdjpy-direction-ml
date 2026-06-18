import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_and_align_usdjpy

# === 路徑設定 ===
H1_PATH = 'data/USDJPY_H1.csv'
H4_PATH = 'data/USDJPY_H4.csv'
D1_PATH = 'data/USDJPY_D1.csv'

print("正在載入並對齊 USDJPY 資料...")
df = load_and_align_usdjpy(H1_PATH, H4_PATH, D1_PATH)
print(f"資料筆數: {len(df)}")

feature_cols = [c for c in df.columns if c not in ['target', 'open', 'high', 'low', 'close', 'volume', 'returns']]
X = df[feature_cols]
y = df['target']

# TimeSeriesSplit 驗證
tscv = TimeSeriesSplit(n_splits=5)
accuracies = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Fold {fold+1} 方向準確率: {acc:.4f}")

print(f"\n平均方向準確率: {sum(accuracies)/len(accuracies):.4f}")

# 最後一 fold 特徵重要性
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
importances.head(15).plot(kind='barh')
plt.title('USDJPY Top 15 特徵重要性')
plt.tight_layout()
plt.show()
