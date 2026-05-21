# 1. Import thư viện
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv(
    r"D:\Hoc May\đồ án\chairs.csv",
    encoding='latin1'
)
# Hiển thị dữ liệu
print(df.head())

# Thông tin dữ liệu
print(df.info())

# Kiểm tra giá trị null
print(df.isnull().sum())

# Thống kê mô tả
print(df.describe())

# 3. Xử lý dữ liệu
# Loại bỏ outlier của Sales
df = df[df['Sales'] < df['Sales'].quantile(0.99)]

# Log transform Sales
df['Sales'] = np.log1p(df['Sales'])

# Feature Engineering
df['discount_amount'] = df['Quantity'] * df['Discount']
df['avg_discount_per_item'] = df['Discount'] / (df['Quantity'] + 1)
df['is_discounted'] = (df['Discount'] > 0).astype(int)
df['is_high_discount'] = (df['Discount'] > 0.2).astype(int)

# 4. Trực quan hóa dữ liệu
# Histogram Sales
sns.histplot(df['Sales'], bins=20)
plt.title("Distribution of Sales")
# plt.show()
# Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), annot=True)
# plt.show()
# Boxplot Quantity vs Sales
sns.boxplot(data=df, x='Quantity', y='Sales')
# plt.show()
# 5. Tách dữ liệu X và y
X = df.drop('Sales', axis=1)
y = df['Sales']

# One Hot Encoding
X = pd.get_dummies(X, drop_first=True)

# 6. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)
# 8. Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# 9. Dự đoán
y_pred = model.predict(X_test)

# 10. Đánh giá mô hình
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("===== KẾT QUẢ =====")
print("R2 Score:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

# Adjusted R2
n = X_test.shape[0]
p = X_test.shape[1]
adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
print("Adjusted R2:", adjusted_r2)

# 11. Cross Validation
scores = cross_val_score(
    model,
    X_scaled,
    y,
    cv=5,
    scoring='r2'
)
print("Cross Validation R2 Mean:", scores.mean())
# 12. So sánh giá trị thực tế và dự đoán

result = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
print(result.head(10))

# 13. Visualization
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
# plt.show()