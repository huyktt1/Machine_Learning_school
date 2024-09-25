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

# Dự đoán
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)

print("Dự đoán:", y_predict)
