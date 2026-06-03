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
# 1. LOAD DATA
df = pd.read_csv(
    r"D:\Hoc May\đồ án\chairs.csv",
    encoding='latin1'
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
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)
# 12. TRAIN MODEL
xgb_model.fit(
    X_train,
    y_train
)

xgb_pred = xgb_model.predict(
    X_test
)
# 14. ĐÁNH GIÁ MÔ HÌNH
xgb_r2 = r2_score(
    y_test,
    xgb_pred
)
xgb_mae = mean_absolute_error(
    y_test,
    xgb_pred
)
xgb_rmse = sqrt(
    mean_squared_error(
        y_test,
        xgb_pred
    )

)
print("Tổng số dòng dữ liệu:", len(df))
print("Số dòng train:", len(X_train))
print("Số dòng test:", len(X_test))

print("\n ===== KẾT QUẢ XGBOOST ====")
print(f"R2 Score : {xgb_r2:.4f}")
print(f"MAE      : {xgb_mae:.4f}")
print(f"RMSE     : {xgb_rmse:.4f}")

# 16. BẢNG CÁC DÒNG DỰ ĐOÁN
prediction_df = pd.DataFrame()
prediction_df['Year'] = X_test['Year']
prediction_df['Month'] = X_test['Month']
prediction_df['Day'] = X_test['Day']
prediction_df['Actual Sales'] = y_test.values
prediction_df['Predicted Sales'] = xgb_pred

pd.options.display.float_format = '{:.2f}'.format
print("\n===== CÁC DÒNG DỰ ĐOÁN =====")
print(
    prediction_df[
        [
            'Year',
            'Month',
            'Day',
            'Actual Sales',
            'Predicted Sales'
        ]
    ].head(200).to_string(index=False)
)

# 17. ACTUAL VS PREDICTED
plt.figure(figsize=(12, 6))
plt.plot(
    y_test.values[:50],
    label='Actual Sales'
)
plt.plot(
    xgb_pred[:50],
    label='Predicted Sales'
)
plt.title("XGBoost - Actual vs Predicted")
plt.xlabel("Samples")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()

# 18. SCATTER PLOT
plt.figure(figsize=(8, 8))
plt.scatter(
    y_test,
    xgb_pred,
    alpha=0.5
)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)
plt.title("XGBoost Scatter Plot")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.grid(True)
plt.show()

# TOP 9 THUỘC TÍNH ẢNH HƯỞNG NHẤT
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
})
importance = importance.sort_values(
    by='Importance',
    ascending=False
)
top9 = importance.head(9)
print("\n===== TOP 9 THUỘC TÍNH QUAN TRỌNG NHẤT =====")
print(top9.to_string(index=False))

# 27. BIỂU ĐỒ FEATURE IMPORTANCE
plt.figure(figsize=(10,6))
plt.barh(
    top9['Feature'],
    top9['Importance']
)
plt.title("TOP 9 THUỘC TÍNH ẢNH HƯỞNG ĐẾN DOANH SỐ")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
import time
start = time.time()
xgb_model.fit(X_train, y_train)
train_time = time.time() - start
print(f" \n Thời gian huấn luyện mô hình: {train_time:.4f} giây")
