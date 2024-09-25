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