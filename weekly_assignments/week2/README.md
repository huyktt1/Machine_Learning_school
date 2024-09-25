
# Hồi Quy Tuyến Tính Giữa Chiều Cao và Cân Nặng

## Công nghệ sử dụng
- **Ngôn ngữ lập trình**: Python
- **Thư viện**: NumPy, Matplotlib

## Thuật toán
- **Hồi quy tuyến tính**: Mô hình hồi quy tuyến tính được sử dụng để dự đoán cân nặng dựa trên chiều cao.

## Chi tiết mã nguồn
```python
import numpy as np
import matplotlib.pyplot as plt

# Dữ liệu chiều cao và cân nặng
X = np.array([180, 162, 183, 174, 160, 163, 180, 165, 175, 170, 170, 169,
              168, 175, 169, 171, 155, 158, 175, 165]).reshape(-1, 1)
y = np.array([86, 55, 86.5, 70, 62, 54, 60, 72, 93, 89, 60, 82, 59, 75,
              56, 89, 45, 60, 60, 72]).reshape(-1, 1)

# Thêm cột bias (toàn giá trị 1)
X_bias = np.insert(X, 0, 1, axis=1)

# Tính toán hệ số hồi quy (theta) sử dụng công thức hồi quy tuyến tính
theta = np.linalg.inv(X_bias.T.dot(X_bias)).dot(X_bias.T).dot(y)

# Định nghĩa hàm dự đoán cân nặng dựa trên chiều cao
def predict_weight(height):
    return theta[0] + theta[1] * height

# Vẽ biểu đồ hồi quy tuyến tính
x1 = 150
y1 = predict_weight(x1)[0]
x2 = 190
y2 = predict_weight(x2)[0]

# Vẽ đường hồi quy (đường thẳng màu đỏ)
plt.plot([x1, x2], [y1, y2], 'r-', label='Đường hồi quy tuyến tính')

# Vẽ các điểm dữ liệu chiều cao và cân nặng (chấm tròn xanh)
plt.plot(X, y, 'bo', label='Dữ liệu thực')

# Nhãn và tiêu đề biểu đồ
plt.xlabel('Chiều cao (cm)')
plt.ylabel('Cân nặng (kg)')
plt.title('Chiều cao và cân nặng của sinh viên VLU')

# Hiển thị chú thích
plt.legend()

# Hiển thị biểu đồ
plt.show()
```
# Hiển thị trang web
![Ảnh chụp màn hình 2024-09-25 163653](https://github.com/user-attachments/assets/d3ffd5d0-255c-4c0d-ad8e-b7f23b237047)

