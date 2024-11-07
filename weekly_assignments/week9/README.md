<h2>Chỉ số Đánh Giá Mô Hình</h2>
<table>
  <thead>
    <tr>
      <th>Chỉ số</th>
      <th>Công thức</th>
      <th>Ý nghĩa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Accuracy</td>
      <td>$$\frac{TP + TN}{TP + TN + FP + FN}$$</td>
      <td>Tỷ lệ dự đoán đúng trên tổng số mẫu</td>
    </tr>
    <tr>
      <td>Recall</td>
      <td>$$\frac{TP}{TP + FN}$$</td>
      <td>Tỷ lệ phát hiện đúng các mẫu dương</td>
    </tr>
    <tr>
      <td>Specificity</td>
      <td>$$\frac{TN}{TN + FP}$$</td>
      <td>Tỷ lệ phát hiện đúng các mẫu âm</td>
    </tr>
    <tr>
      <td>Precision</td>
      <td>$$\frac{TP}{TP + FP}$$</td>
      <td>Tỷ lệ các dự đoán dương chính xác</td>
    </tr>
    <tr>
      <td>F1-Score</td>
      <td>$$\frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$</td>
      <td>Cân bằng giữa Precision và Recall</td>
    </tr>
  </tbody>
</table>

<h2>Ma Trận Nhầm Lẫn và Tính Toán Các Chỉ Số</h2>
<pre><code>
TN = 50
FP = 10
FN = 5
TP = 30

# Tính toán các chỉ số
accuracy = (TP + TN) / (TP + TN + FP + FN)
recall = TP / (TP + FN)
specificity = TN / (TN + FP)
precision = TP / (TP + FP)
f1 = 2 * (precision * recall) / (precision + recall)

result_1 = {
    "Confusion Matrix": [[TN, FP], [FN, TP]],
    "Accuracy": accuracy,
    "Recall": recall,
    "Specificity": specificity,
    "Precision": precision,
    "F1 Score": f1
}

print(result_1)
</code></pre>

<h2>Các Chỉ Số Bổ Sung</h2>
<table>
  <thead>
    <tr>
      <th>Chỉ số</th>
      <th>Công thức</th>
      <th>Ý nghĩa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Balanced Accuracy</td>
      <td>$$\frac{\text{TPR} + \text{TNR}}{2}$$</td>
      <td>Đánh giá độ chính xác trung bình của mô hình với cả hai lớp, tránh ảnh hưởng của dữ liệu mất cân bằng.</td>
    </tr>
    <tr>
      <td>Matthews Correlation Coefficient (MCC)</td>
      <td>$$\frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$</td>
      <td>Đo lường mối tương quan giữa các giá trị dự đoán và thực tế, với giá trị từ -1 đến 1, chỉ ra mô hình tốt hơn khi gần 1.</td>
    </tr>
    <tr>
      <td>Fowlkes-Mallows Index (FMI)</td>
      <td>$$\sqrt{\frac{TP}{TP + FP} \times \frac{TP}{TP + FN}}$$</td>
      <td>Đo lường độ chính xác của mô hình dựa trên độ nhạy và độ chính xác của các dự đoán dương tính.</td>
    </tr>
    <tr>
      <td>Bias</td>
      <td>$$\frac{FP + FN}{TP + TN + FP + FN}$$</td>
      <td>Đánh giá mức độ sai lệch của mô hình, thể hiện tỷ lệ dự đoán sai so với tổng số dự đoán.</td>
    </tr>
  </tbody>
</table>

<pre><code>
# Định nghĩa ma trận nhầm lẫn và tính toán các chỉ số bổ sung
TN = 50
FP = 10
FN = 5
TP = 30

accuracy = (TP + TN) / (TP + TN + FP + FN)
balanced_accuracy = (recall + specificity) / 2
mcc = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
fmi = ((TP / (TP + FP)) * (TP / (TP + FN))) ** 0.5
bias = (FP + FN) / (TP + TN + FP + FN)

result_2 = {
    "Confusion Matrix": [[TN, FP], [FN, TP]],
    "Balanced Accuracy": balanced_accuracy,
    "MCC": mcc,
    "FMI": fmi,
    "Bias": bias
}

print(result_2)
</code></pre>
--------------------------------------
<h1>Wine Classification with K-Nearest Neighbors (KNN)</h1>

<h2>Import các thư viện cần thiết</h2>
<pre><code>import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
</code></pre>
<p>- <code>numpy</code> và <code>pandas</code>: Thư viện để xử lý dữ liệu.</p>
<p>- <code>sklearn.datasets</code>: Dùng để tải bộ dữ liệu mẫu (ở đây là dữ liệu về rượu vang).</p>
<p>- <code>train_test_split</code>: Chia dữ liệu thành tập huấn luyện và tập kiểm tra.</p>
<p>- <code>KNeighborsClassifier</code>: Mô hình KNN để phân loại.</p>
<p>- <code>accuracy_score</code>, <code>recall_score</code>, <code>precision_score</code>: Các hàm đánh giá hiệu suất của mô hình.</p>

<h2>Tải tập dữ liệu Wine</h2>
<pre><code>data = load_wine()
X = data.data
y = data.target
</code></pre>
<p>- <code>load_wine()</code>: Tải bộ dữ liệu về rượu vang.</p>
<p>- <code>X</code>: Chứa các đặc trưng của mẫu dữ liệu.</p>
<p>- <code>y</code>: Chứa nhãn tương ứng của từng mẫu.</p>

<h2>Chia tập dữ liệu theo tỷ lệ 70:30</h2>
<pre><code>X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
</code></pre>
<p>- <code>test_size=0.3</code>: Chia 30% dữ liệu cho tập kiểm tra, 70% cho tập huấn luyện.</p>
<p>- <code>random_state=42</code>: Đảm bảo chia dữ liệu ngẫu nhiên có thể tái lập.</p>

<h2>Khởi tạo mô hình KNN với k=5</h2>
<pre><code>k = 5
knn = KNeighborsClassifier(n_neighbors=k)
</code></pre>
<p>- <code>n_neighbors=5</code>: Số lượng hàng xóm gần nhất trong mô hình KNN.</p>

<h2>Huấn luyện mô hình trên tập huấn luyện</h2>
<pre><code>knn.fit(X_train, y_train)
</code></pre>
<p>- <code>fit</code>: Huấn luyện mô hình KNN trên tập dữ liệu huấn luyện.</p>

<h2>Dự đoán nhãn trên tập kiểm tra</h2>
<pre><code>y_pred = knn.predict(X_test)
</code></pre>
<p>- <code>predict</code>: Dự đoán nhãn của tập kiểm tra dựa trên mô hình đã huấn luyện.</p>

<h2>Tính toán độ chính xác, recall và precision</h2>
<pre><code>accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
</code></pre>
<p>- <code>accuracy_score</code>: Tính độ chính xác của mô hình.</p>
<p>- <code>recall_score</code> và <code>precision_score</code>: Đánh giá độ nhớ và độ chính xác của mô hình.</p>

<h2>In ra kết quả</h2>
<pre><code>print(f"Độ chính xác (Accuracy): {accuracy:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Precision: {precision:.2f}")
</code></pre>
<p>- In các kết quả độ chính xác, recall và precision.</p>
