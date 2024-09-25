# Dự đoán Hồi Quy Tuyến Tính

## Công nghệ sử dụng
- **Ngôn ngữ lập trình**: Python
- **Thư viện**: NumPy

## Thuật toán
- **Hồi quy tuyến tính**: Sử dụng phương pháp bình phương tối thiểu để tìm trọng số (theta) tối ưu cho mô hình hồi quy.

## Chi tiết mã nguồn
```python
# Bài toán 1:
import numpy as np

# Khởi tạo seed cho kết quả ngẫu nhiên
np.random.seed(0)

# Tạo dữ liệu ngẫu nhiên
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Thêm bias vào dữ liệu
X_b = np.c_[np.ones((100, 1)), X]

# Tính toán trọng số tối ưu
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print("Trọng số (theta):", theta_best)
```

# Dự đoán Giá Bất Động Sản Dựa Trên Diện Tích

## Công nghệ sử dụng
- **Ngôn ngữ lập trình**: Python
- **Thư viện**: NumPy, Pandas, Matplotlib

## Thuật toán
- **Hồi quy tuyến tính**: Sử dụng phương pháp bình phương tối thiểu để ước lượng giá bất động sản dựa trên diện tích.

## Chi tiết mã nguồn
```python
# Bài toán 2:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data = pd.read_csv('/content/linear.csv')

# Tách dữ liệu thành đặc trưng (X) và mục tiêu (y)
X = data['Diện tích'].values.reshape(-1, 1)
y = data['Giá'].values.reshape(-1, 1)

# Thêm bias vào dữ liệu
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Tính toán trọng số tối ưu
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Dự đoán
y_predict = X_b.dot(theta_best)

# Hàm tính RMSE
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Tính toán RMSE
loss_rmse = rmse(y, y_predict)
print(f"RMSE: {loss_rmse}")

# Vẽ biểu đồ
plt.figure(figsize=(8, 6))
plt.plot(X, y, "b.", label="Dữ liệu")
plt.plot(X, y_predict, "r-", label="Dự đoán (Hồi quy tuyến tính)")
plt.xlabel("Diện tích")
plt.ylabel("Giá")
plt.title(f"Fit Hồi Quy Tuyến Tính (RMSE: {loss_rmse:.4f})")
plt.legend()
plt.grid(True)
plt.show()
# Dự đoán
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)

print("Dự đoán:", y_predict)
```
# Dự đoán Giá Bất Động Sản Dựa Trên Diện Tích (Hồi Quy Đa Thức)

## Công nghệ sử dụng
- **Ngôn ngữ lập trình**: Python
- **Thư viện**: NumPy, Pandas, Matplotlib

## Thuật toán
- **Hồi quy đa thức**: Sử dụng hồi quy bậc hai để dự đoán giá bất động sản dựa trên diện tích.

## Chi tiết mã nguồn
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data = pd.read_csv('/content/linear.csv')

# Tách dữ liệu thành đặc trưng (X) và mục tiêu (y)
X = data['Diện tích'].values.reshape(-1, 1)
y = data['Giá'].values.reshape(-1, 1)

# Tạo đặc trưng đa thức (X^2)
X_poly = np.c_[np.ones((X.shape[0], 1)), X, X**2]  # Thêm bias (1), X, và X^2

# Tính toán trọng số (theta) bằng phương pháp bình phương tối thiểu
theta_best = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)

# Dự đoán các giá trị sử dụng mô hình parabol
y_predict = X_poly.dot(theta_best)

# Định nghĩa hàm RMSE
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Tính toán RMSE
loss_rmse = rmse(y, y_predict)
print(f"RMSE: {loss_rmse}")

# Vẽ biểu đồ các điểm dữ liệu và đường dự đoán (hồi quy parabol)
plt.figure(figsize=(8, 6))
plt.plot(X, y, "b.", label="Dữ liệu")
plt.plot(X, y_predict, "r-", label="Dự đoán (Hồi quy đa thức)")
plt.xlabel("Diện tích")
plt.ylabel("Giá")
plt.title(f"Fit Hồi Quy Đa Thức (RMSE: {loss_rmse:.4f})")
plt.legend()
plt.grid(True)
plt.show()
```
# Dự Đoán Khối Lượng Cơ Thể (Kg) Dựa Trên Các Đặc Trưng

## Công nghệ sử dụng
- **Ngôn ngữ lập trình**: Python
- **Thư viện**: NumPy, Pandas, Matplotlib

## Thuật toán
- **Hồi quy tuyến tính**: Sử dụng hồi quy tuyến tính đa biến để dự đoán khối lượng cơ thể dựa trên các đặc trưng như cân nặng, cổ chân, hoạt động và % mỡ.

## Chi tiết mã nguồn
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data = pd.read_csv('/content/sample.csv')

# Chọn các cột liên quan cho đặc trưng và mục tiêu
X = data[['pound', 'Femoral Neck', 'Activity', '%Fat S']].values
y = data['Kg'].values.reshape(-1, 1)

# Thêm bias (cột một) vào X
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Tính toán trọng số (theta) bằng phương pháp bình phương tối thiểu
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Dự đoán các giá trị sử dụng mô hình
y_predict = X_b.dot(theta_best)

# Định nghĩa hàm RMSE
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Tính toán RMSE
loss_rmse = rmse(y, y_predict)
print(f"RMSE ban đầu với tất cả các đặc trưng: {loss_rmse}")

# Vẽ biểu đồ giá trị thực và giá trị dự đoán
plt.figure(figsize=(8, 6))
plt.plot(y, "b.", label="Khối lượng thực (Kg)")
plt.plot(y_predict, "r-", label="Khối lượng dự đoán (Hồi quy tuyến tính)")
plt.xlabel("Chỉ số")
plt.ylabel("Kg")
plt.title(f"Dự Đoán so với Thực Tế (RMSE: {loss_rmse:.4f})")
plt.legend()
plt.grid(True)
plt.show()
