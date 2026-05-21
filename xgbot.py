import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from xgboost import XGBRegressor
from math import sqrt

# ==========================================================
# 1. LOAD DATA
# ==========================================================
df = pd.read_csv(
    r"D:\Hoc May\đồ án\chairs.csv",
    encoding='latin1'
)

# ==========================================================
# 2. CHỌN ĐẶC TRƯNG
# ==========================================================
selected_columns = [
    'Order Date',
    'Ship Mode',
    'Segment',
    'Region',
    'Quantity',
    'Discount',
    'Profit',
    'Category',
    'Sub-Category',
    'Sales'
]

df = df[selected_columns].copy()

# ==========================================================
# 3. XỬ LÝ DỮ LIỆU THIẾU
# ==========================================================
df.dropna(inplace=True)

# ==========================================================
# 4. LOẠI OUTLIER
# ==========================================================
df = df[df['Sales'] < df['Sales'].quantile(0.99)]

# ==========================================================
# 5. FEATURE ENGINEERING
# ==========================================================
df['Order Date'] = pd.to_datetime(df['Order Date'])

df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['DayOfWeek'] = df['Order Date'].dt.dayofweek

df.drop('Order Date', axis=1, inplace=True)

# ==========================================================
# 6. TARGET TRANSFORMATION
# ==========================================================
y = np.log1p(df['Sales'])

# ==========================================================
# 7. FEATURES
# ==========================================================
X = df.drop('Sales', axis=1)

# One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)

print("Số đặc trưng:", X.shape[1])

# ==========================================================
# 8. TRAIN TEST SPLIT
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# 9. XGBOOST MODEL
# ==========================================================
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# ==========================================================
# 10. PREDICTION
# ==========================================================
y_pred_log = model.predict(X_test)

# Chuyển về giá trị thật
y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred_log)

# ==========================================================
# 11. EVALUATION
# ==========================================================
r2 = r2_score(y_test_real, y_pred_real)

mae = mean_absolute_error(
    y_test_real,
    y_pred_real
)

rmse = sqrt(mean_squared_error(
    y_test_real,
    y_pred_real
))

print("\n===== KẾT QUẢ =====")
print(f"R2 Score : {r2:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")

# ==========================================================
# 12. SO SÁNH DỰ ĐOÁN
# ==========================================================
results = pd.DataFrame({
    'Actual Sales': y_test_real.values[:10],
    'Predicted Sales': y_pred_real[:10]
})

print("\n===== 10 GIÁ TRỊ DỰ ĐOÁN =====")
print(results)

# ==========================================================
# 13. FEATURE IMPORTANCE
# ==========================================================
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\n===== TOP FEATURE =====")
print(importance.head(10))

# ==========================================================
# 14. VẼ BIỂU ĐỒ
# ==========================================================
top10 = importance.head(10)

plt.figure(figsize=(10,6))

plt.barh(
    top10['Feature'],
    top10['Importance']
)

plt.title("Top 10 Important Features")
plt.xlabel("Importance")

plt.gca().invert_yaxis()

plt.show()