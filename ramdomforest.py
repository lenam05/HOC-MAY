import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#B1: doc du lieu dataset
df = pd.read_csv(
    r"D:\Hoc May\đồ án\chairs.csv",
    encoding='latin1'
)

print("Kích thước dataset:", df.shape)
# B2: LÀM SẠCH DỮ LIỆU
print("\nDữ liệu thiếu:")
print(df.isnull().sum())


df = df.dropna()


df = df.drop_duplicates()

print("\nKích thước sau khi làm sạch:", df.shape)


# xu ly order date

df['Order Date'] = pd.to_datetime(df['Order Date'])

# tach ngay thang nam
df['Order_Day'] = df['Order Date'].dt.day
df['Order_Month'] = df['Order Date'].dt.month
df['Order_Year'] = df['Order Date'].dt.year

# B3: phan tich du lieu kham pha eda
print("\nThông tin dữ liệu:")
print(df.info())

print("\nThống kê dữ liệu:")
print(df.describe())

# Histogram doanh số
plt.figure(figsize=(8,5))

plt.hist(
    df['Sales'],
    bins=30
)

plt.title("Distribution of Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.show()

# LOG TRANSFORM TARGET
y = np.log1p(df['Sales'])

# INPUT FEATURES
X = df[[
    'Category',
    'Sub-Category',
    'Quantity',
    'Discount',
    'Profit',
    'Ship Mode',
    'Segment',
    'Region',
    'Order_Day',
    'Order_Month',
    'Order_Year'
]]

# ONE HOT ENCODING
X = pd.get_dummies(
    X,
    drop_first=True
)

print("\nSố lượng features sau encoding:", X.shape[1])
# B4: CHIA TRAIN VÀ TEST
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# RANDOM FOREST REGRESSOR
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# B5: HUẤN LUYỆN MÔ HÌNH
model.fit(X_train, y_train)

# B6: ĐÁNH GIÁ MÔ HÌNH
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Dự đoán
y_pred = model.predict(X_test)

# Chuyển ngược log transform
y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred)

# TÍNH TOÁN METRICS
mae = mean_absolute_error(
    y_test_real,
    y_pred_real
)

rmse = np.sqrt(
    mean_squared_error(
        y_test_real,
        y_pred_real
    )
)

r2 = r2_score(
    y_test_real,
    y_pred_real
)

print("\n===== MODEL EVALUATION =====")

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)
# FEATURE IMPORTANCE

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop 10 Important Features:")
print(importance.head(10))
# BIỂU ĐỒ FEATURE IMPORTANCE
top10 = importance.head(10)

plt.figure(figsize=(10,6))

plt.barh(
    top10['Feature'],
    top10['Importance']
)

plt.title("Top 10 Feature Importance")
plt.xlabel("Importance")

plt.gca().invert_yaxis()

plt.show()


# B7: DỰ ĐOÁN
sample_prediction = y_pred_real[:5]

print("\nDự đoán doanh số 5 mẫu đầu tiên:")
print(sample_prediction)