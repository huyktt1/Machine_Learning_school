# WEEEK 1:
1) CODE MACHINE LEARNING :
# problem 1:
import numpy as np


np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X]

theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print("Trọng số (theta):", theta_best)

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)

print("Dự đoán:", y_predict)
------------------------------------
Trọng số (theta): [[4.22215108]
 [2.96846751]]
Dự đoán: [[ 4.22215108]
 [10.1590861 ]]

# problem 2:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('/content/linear.csv')

X = data['Diện tích'].values.reshape(-1, 1)
y = data['Giá'].values.reshape(-1, 1)

X_b = np.c_[np.ones((X.shape[0], 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
y_predict = X_b.dot(theta_best)
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

loss_rmse = rmse(y, y_predict)
print(f"RMSE: {loss_rmse}")

plt.figure(figsize=(8, 6))
plt.plot(X, y, "b.", label="Data points")
plt.plot(X, y_predict, "r-", label="Prediction (Linear Regression)")
plt.xlabel("X")
plt.ylabel("y")
plt.title(f"Linear Regression Fit (RMSE: {loss_rmse:.4f})")
plt.legend()
plt.grid(True)
plt.show()

# problem 3:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/content/linear.csv')

X = data['Diện tích'].values.reshape(-1, 1)
y = data['Giá'].values.reshape(-1, 1)


X_poly = np.c_[np.ones((X.shape[0], 1)), X, X**2]  # Adding bias (1), X, and X^2


theta_best = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)


y_predict = X_poly.dot(theta_best)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


loss_rmse = rmse(y, y_predict)
print(f"RMSE: {loss_rmse}")

plt.figure(figsize=(8, 6))
plt.plot(X, y, "b.", label="Data points")
plt.plot(X, y_predict, "r-", label="Prediction (Parabolic Regression)")
plt.xlabel("X")
plt.ylabel("y")
plt.title(f"Parabolic Regression Fit (RMSE: {loss_rmse:.4f})")
plt.legend()
plt.grid(True)
plt.show()
------------------------------------------------------------------------------------------------------------------------
# week2 :
# machine learning code include flask.
from flask import Flask, render_template
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    # Dữ liệu chiều cao và cân nặng
    X = np.array([180, 162, 183, 174, 160, 163, 180, 165, 175, 170, 170, 169,
                  168, 175, 169, 171, 155, 158, 175, 165]).reshape(-1, 1)
    y = np.array([86, 55, 86.5, 70, 62, 54, 60, 72, 93, 89, 60, 82, 59, 75,
                  56, 89, 45, 60, 60, 72]).reshape(-1, 1)

    # Thêm cột bias (toàn giá trị 1)
    X_bias = np.insert(X, 0, 1, axis=1)

    # Tính toán hệ số hồi quy (theta)
    theta = np.linalg.inv(X_bias.T.dot(X_bias)).dot(X_bias.T).dot(y)

    # Hàm dự đoán cân nặng
    def predict_weight(height):
        return theta[0] + theta[1] * height

    # Vẽ biểu đồ hồi quy tuyến tính
    x1 = 150
    y1 = predict_weight(x1)[0]
    x2 = 190
    y2 = predict_weight(x2)[0]

    plt.plot([x1, x2], [y1, y2], 'r-', label='Đường hồi quy tuyến tính')
    plt.plot(X, y, 'bo', label='Dữ liệu thực')
    plt.xlabel('Chiều cao (cm)')
    plt.ylabel('Cân nặng (kg)')
    plt.title('Chiều cao và cân nặng của sinh viên VLU')
    plt.legend()

    # Lưu biểu đồ vào bộ nhớ
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('index.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)

# 2) BASIC WEB CODE:

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kết quả Hồi Quy Tuyến Tính</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            color: #333;
        }
        .container {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
            max-width: 600px;
            width: 100%;
            text-align: center;
        }
        h1 {
            color: #4CAF50;
            margin-bottom: 20px;
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
        }
        footer {
            margin-top: 20px;
            font-size: 14px;
            color: #777;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Kết quả Hồi Quy Tuyến Tính</h1>
        <img src="data:image/png;base64,{{ plot_url }}" alt="Đường hồi quy tuyến tính">
        <footer>
            © 2024 - Machine Learning Results
        </footer>
    </div>
</body>
</html>
