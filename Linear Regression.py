import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.linear_model import LinearRegression
from math import sqrt

output_dir = r"D:\Hoc May\đồ án"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. LOAD DATA
df = pd.read_excel(
    r"c:\Users\84378\Downloads\stores_sales_forecasting17.xlsx",
    engine='openpyxl'
)

# 2. CHỌN CỘT DỮ LIỆU
selected_columns = [
    'Order Date',
    'Ship Mode',
    'Segment',
    'Region',
    'Category',
    'Sub-Category',
    'Quantity',
    'Discount',
    'Profit',
    'Sales'
]
df = df[selected_columns].copy()

# 3. XỬ LÝ DỮ LIỆU THIẾU
df.dropna(inplace=True)

# 4. XÓA OUTLIER
df = df[df['Sales'] < df['Sales'].quantile(0.99)]

# 5. CHUYỂN KIỂU NGÀY THÁNG
df['Order Date'] = pd.to_datetime(df['Order Date'])

# 6. TÁCH ĐẶC TRƯNG THỜI GIAN
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Day'] = df['Order Date'].dt.day
df['DayOfWeek'] = df['Order Date'].dt.dayofweek
df['Quarter'] = df['Order Date'].dt.quarter
y = df['Sales']

# 8. FEATURES
X = df.drop(
    ['Sales', 'Order Date'],
    axis=1
)

# 9. ONE HOT ENCODING
X = pd.get_dummies(
    X,
    drop_first=True
)

print("\nSố đặc trưng sau encoding:", X.shape[1])

# 10. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# KHỞI TẠO MÔ HÌNH LINEAR REGRESSION
lr_model = LinearRegression()

# 12. TRAIN MODEL
lr_model.fit(
    X_train,
    y_train
)

# 13. PREDICT
lr_pred = lr_model.predict(
    X_test
)

# 14. ĐÁNH GIÁ MÔ HÌNH
lr_r2 = r2_score(
    y_test,
    lr_pred
)
lr_mae = mean_absolute_error(
    y_test,
    lr_pred
)
lr_rmse = sqrt(
    mean_squared_error(
        y_test,
        lr_pred
    )
)

print("\n" + "=" * 50)
print("KẾT QUẢ LINEAR REGRESSION")
print("=" * 50)
print(f"R2 Score : {lr_r2:.4f}")
print(f"MAE      : {lr_mae:.4f}")
print(f"RMSE     : {lr_rmse:.4f}")

# 16. BẢNG 20 DÒNG DỰ ĐOÁN
prediction_df = pd.DataFrame()
prediction_df['Year'] = X_test['Year']
prediction_df['Month'] = X_test['Month']
prediction_df['Day'] = X_test['Day']
prediction_df['Actual Sales'] = y_test.values
prediction_df['Predicted Sales'] = lr_pred

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.options.display.float_format = '{:.2f}'.format

print("\n===== 20 DÒNG DỰ ĐOÁN =====")
print(
    prediction_df[
        [
            'Year',
            'Month',
            'Day',
            'Actual Sales',
            'Predicted Sales'
        ]
    ].head(20).to_string(index=False)
)

# HÌNH 1: ĐỒ THỊ ĐƯỜNG ACTUAL VS PREDICTED (MẪU 50 DÒNG ĐẦU)
plt.figure(figsize=(12, 6))
plt.plot(
    y_test.values[:50],
    label='Actual'
)
plt.plot(
    lr_pred[:50],
    label='Predicted'
)
plt.title("Linear Regression")
plt.xlabel("Samples")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "linear_regression_line_plot.png"), dpi=300)
plt.show()

# HÌNH 2: BIỂU ĐỒ PHÂN TÁN (SCATTER PLOT)
plt.figure(figsize=(8, 8))
plt.scatter(
    y_test,
    lr_pred,
    alpha=0.5
)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)
plt.title("Linear Regression Scatter Plot")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "linear_regression_scatter_plot.png"), dpi=300)
plt.show()

# HÌNH 3: BIỂU ĐỒ THUỘC TÍNH QUAN TRỌNG (TOP 10 FEATURE IMPORTANCE)
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(lr_model.coef_)
})
importance = importance.sort_values(
    by='Importance',
    ascending=False
)
top10 = importance.head(10)

print("\n===== TOP 10 THUỘC TÍNH QUAN TRỌNG NHẤT =====")
print(top10.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(
    top10['Feature'],
    top10['Importance']
)
plt.title("Top 10 Important Features")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "linear_regression_feature_importance.png"), dpi=300)
plt.show()
import time
start = time.time()
lr_model.fit(X_train, y_train)
train_time = time.time() - start
print(f"Training Time: {train_time:.4f} seconds") 
