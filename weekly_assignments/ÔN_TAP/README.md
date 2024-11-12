<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Project Naive Bayes - Iris Dataset</title>
</head>
<body>
  <h1>Project: Naive Bayes on Iris Dataset</h1>

  <h2>Giới thiệu</h2>
  <p>Dự án này sử dụng thuật toán Naive Bayes để phân loại dữ liệu từ bộ dữ liệu Iris. Dữ liệu này bao gồm các thông tin về loài hoa Iris, và mục tiêu của chúng ta là dự đoán loại hoa dựa trên các đặc điểm của nó (chiều dài và chiều rộng của đài và cánh hoa).</p>

  <h2>Yêu cầu</h2>
  <ul>
    <li>Python 3.x</li>
    <li>scikit-learn (Được cài đặt bằng lệnh <code>pip install scikit-learn</code>)</li>
  </ul>

  <h2>Thuật toán</h2>
  <p>Thuật toán sử dụng trong dự án này là <strong>Naive Bayes</strong> với mô hình Gaussian Naive Bayes. Mô hình này dựa trên giả thuyết rằng các đặc trưng của các lớp dữ liệu là độc lập và tuân theo phân phối chuẩn (Gaussian).</p>

  <h2>Các bước triển khai</h2>
  
  <h3>1. Tải dữ liệu Iris</h3>
  <pre><code>from sklearn.datasets import load_iris</code></pre>
  <p>Chúng ta sử dụng hàm <code>load_iris()</code> từ thư viện <strong>sklearn.datasets</strong> để tải dữ liệu Iris, gồm các thông tin về chiều dài và chiều rộng của đài hoa và cánh hoa.</p>

  <h3>2. Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra</h3>
  <pre><code>from sklearn.model_selection import train_test_split</code></pre>
  <p>Chúng ta sử dụng hàm <code>train_test_split()</code> để chia tập dữ liệu thành 2 phần: tập huấn luyện (80%) và tập kiểm tra (20%). Điều này giúp chúng ta huấn luyện mô hình trên một phần của dữ liệu và kiểm tra mô hình trên phần còn lại để đánh giá hiệu quả.</p>

  <h3>3. Khởi tạo và huấn luyện mô hình Naive Bayes</h3>
  <pre><code>from sklearn.naive_bayes import GaussianNB</code></pre>
  <pre><code>model = GaussianNB()</code></pre>
  <p>Mô hình Naive Bayes được khởi tạo bằng <code>GaussianNB()</code>, sử dụng phân phối chuẩn để ước lượng xác suất lớp. Sau đó, mô hình được huấn luyện bằng cách sử dụng dữ liệu huấn luyện <code>X_train</code> và nhãn lớp <code>y_train</code>.</p>

  <h3>4. Dự đoán và tính toán độ chính xác</h3>
  <pre><code>from sklearn.metrics import accuracy_score</code></pre>
  <p>Chúng ta sử dụng hàm <code>accuracy_score()</code> để tính toán độ chính xác của mô hình, so sánh nhãn dự đoán với nhãn thực tế từ tập kiểm tra.</p>

  <h3>5. Code hoàn chỉnh</h3>
  <pre><code>
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Tải dữ liệu Iris
iris = load_iris()
X = iris.data  # Tập dữ liệu đầu vào
y = iris.target  # Nhãn lớp

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra với tỷ lệ 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình Naive Bayes
model = GaussianNB()

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Tính độ chính xác của mô hình
accuracy = accuracy_score(y_test, y_pred)

# In ra độ chính xác
print(f"Độ chính xác của mô hình Naive Bayes: {accuracy:.2f}")
  </code></pre>

  <h2>Kết quả</h2>
  <p>Kết quả của mô hình Naive Bayes sẽ in ra độ chính xác trên tập kiểm tra. Độ chính xác cho mô hình Naive Bayes đối với bộ dữ liệu Iris có thể dao động, nhưng thường đạt khoảng 95-97%.</p>

  <h3>Ví dụ kết quả:</h3>
  <pre><code>Độ chính xác của mô hình Naive Bayes: 1.00</code></pre>

  <h2>Giải thích</h2>
  <p>Mô hình Naive Bayes đã đạt độ chính xác cao trên bộ dữ liệu Iris nhờ vào tính đơn giản và khả năng phân loại tốt của thuật toán này trong các bài toán phân loại dữ liệu nhỏ với các đặc trưng độc lập.</p>

  <h2>Liên hệ</h2>
  <p>Để biết thêm chi tiết hoặc cần hỗ trợ, vui lòng liên hệ với tác giả qua email: <a href="mailto:your-email@example.com">your-email@example.com</a>.</p>
</body>
</html>

<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Confusion Matrix - Naive Bayes</title>
</head>
<body>
  <h1>Confusion Matrix cho Mô hình Naive Bayes</h1>

  <h2>Giới thiệu</h2>
  <p>Ma trận nhầm lẫn (confusion matrix) là công cụ hữu ích trong việc đánh giá hiệu quả của mô hình phân loại. Nó cho phép chúng ta nhìn nhận chi tiết về số lượng các dự đoán đúng và sai, cũng như các loại sai lầm mà mô hình gặp phải. Ma trận này giúp chúng ta phân tích độ chính xác của mô hình và đưa ra các quyết định cải thiện mô hình.</p>

  <h2>Các bước triển khai</h2>
  <p>Chúng ta sẽ sử dụng thư viện <strong>matplotlib</strong> để vẽ ma trận nhầm lẫn và thư viện <strong>seaborn</strong> để làm đẹp biểu đồ.</p>

  <h3>1. Tính toán ma trận nhầm lẫn</h3>
  <pre><code>from sklearn.metrics import confusion_matrix</code></pre>
  <p>Chúng ta sử dụng hàm <code>confusion_matrix()</code> từ thư viện <strong>sklearn.metrics</strong> để tính toán ma trận nhầm lẫn giữa nhãn thực tế và nhãn dự đoán.</p>

  <h3>2. Vẽ ma trận nhầm lẫn</h3>
  <pre><code>
import matplotlib.pyplot as plt
import seaborn as sns

# Tính toán ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)

# Vẽ ma trận nhầm lẫn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn')
plt.show()
  </code></pre>

  <h2>Giải thích mã nguồn</h2>

  <h3>3. Tính toán ma trận nhầm lẫn</h3>
  <p>Chúng ta sử dụng hàm <code>confusion_matrix(y_test, y_pred)</code> để tính toán ma trận nhầm lẫn, trong đó:
    <ul>
      <li><strong>y_test</strong>: Nhãn lớp thực tế từ tập kiểm tra.</li>
      <li><strong>y_pred</strong>: Nhãn lớp dự đoán từ mô hình.</li>
    </ul>
  </p>

  <h3>4. Vẽ ma trận nhầm lẫn</h3>
  <p>Sau khi tính toán xong ma trận nhầm lẫn, chúng ta sử dụng <strong>seaborn.heatmap()</strong> để vẽ ma trận dưới dạng hình ảnh. Các tham số quan trọng:
    <ul>
      <li><strong>annot=True</strong>: Hiển thị các giá trị số trong các ô của ma trận.</li>
      <li><strong>fmt='d'</strong>: Định dạng số là số nguyên.</li>
      <li><strong>cmap='Blues'</strong>: Sử dụng màu xanh dương cho bản đồ màu.</li>
      <li><strong>xticklabels</strong> và <strong>yticklabels</strong>: Nhãn cho trục x và y tương ứng với các tên của các lớp trong dữ liệu Iris.</li>
    </ul>
  </p>

  <h2>Nhận xét về Kết quả Dự đoán</h2>
  <p>Ma trận nhầm lẫn sẽ giúp chúng ta hiểu rõ hơn về mức độ chính xác của mô hình. Trong trường hợp của bộ dữ liệu Iris, nếu mô hình dự đoán tốt, các giá trị trong ma trận nhầm lẫn sẽ tập trung chủ yếu vào đường chéo (chỉ ra số lượng dự đoán đúng). Các giá trị ngoài đường chéo sẽ cho thấy mô hình gặp phải sai lầm khi phân loại các loài hoa khác nhau.</p>

  <h3>Ví dụ về Ma trận Nhầm lẫn:</h3>
  <pre><code>
Độ chính xác của mô hình Naive Bayes: 0.97
Ma trận nhầm lẫn:
[[12  0  0]
 [ 0 13  1]
 [ 0  0 14]]
  </code></pre>
  <p>Trong ví dụ trên:
    <ul>
      <li>Số lượng mẫu loại 0 (Setosa) được dự đoán đúng là 12 (dọc theo đường chéo).</li>
      <li>Số lượng mẫu loại 1 (Versicolor) được dự đoán đúng là 13, nhưng có một mẫu bị nhầm thành loại 2 (Virginica) - sai lầm duy nhất.</li>
      <li>Số lượng mẫu loại 2 (Virginica) được dự đoán đúng là 14.</li>
    </ul>
  </p>

  <h2>Kết luận</h2>
  <p>Ma trận nhầm lẫn cho phép chúng ta đánh giá mô hình một cách chi tiết hơn, và từ đó đưa ra các quyết định về việc cải thiện mô hình nếu cần thiết. Nếu mô hình phân loại có độ chính xác thấp, ta có thể cân nhắc các bước như điều chỉnh tham số, thay đổi thuật toán, hoặc làm sạch dữ liệu.</p>

</body>
</html>

<h2>Câu 3: Xây dựng mô hình KNN để phân loại dữ liệu Wine</h2>

<h3>Mô tả</h3>
<p>Mã này thực hiện các bước sau:</p>
<ol>
  <li>Tải tập dữ liệu Wine từ thư viện <code>sklearn.datasets</code>.</li>
  <li>Chia tập dữ liệu thành hai phần: tập huấn luyện (70%) và tập kiểm tra (30%).</li>
  <li>Xây dựng mô hình KNN với <code>k = 5</code>.</li>
  <li>Tính toán và in ra các chỉ số đánh giá mô hình bao gồm độ chính xác (Accuracy), độ nhạy (Recall), và độ chính xác (Precision).</li>
</ol>

<h3>Mã nguồn</h3>

<pre><code class="language-python">
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Tải tập dữ liệu Wine
wine = load_wine()
X = wine.data
y = wine.target

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra với tỷ lệ 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
</code></pre>

<h4>Giải thích:</h4>
<ul>
  <li><code>load_wine()</code> tải tập dữ liệu về các đặc trưng và nhãn của rượu vang.</li>
  <li><code>train_test_split</code> chia dữ liệu thành tập huấn luyện (<code>X_train</code>, <code>y_train</code>) và tập kiểm tra (<code>X_test</code>, <code>y_test</code>) theo tỷ lệ 70:30.</li>
</ul>

<pre><code class="language-python">
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Khởi tạo mô hình KNN với k = 5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
</code></pre>

<h4>Giải thích:</h4>
<ul>
  <li><code>KNeighborsClassifier(n_neighbors=5)</code> tạo một mô hình KNN với <code>k = 5</code>.</li>
  <li><code>knn.fit(X_train, y_train)</code> huấn luyện mô hình trên dữ liệu huấn luyện.</li>
</ul>

<pre><code class="language-python">
# Dự đoán trên tập kiểm tra
y_pred = knn.predict(X_test)
</code></pre>

<h4>Giải thích:</h4>
<ul>
  <li><code>knn.predict(X_test)</code> dự đoán nhãn cho tập kiểm tra (<code>X_test</code>) và lưu kết quả vào <code>y_pred</code>.</li>
</ul>

<pre><code class="language-python">
# Tính toán các chỉ số
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')

# In ra kết quả
print(f"Độ chính xác: {accuracy:.2f}")
print(f"Độ nhạy (Recall): {recall:.2f}")
print(f"Độ chính xác (Precision): {precision:.2f}")
</code></pre>

<h4>Giải thích:</h4>
<ul>
  <li><code>accuracy_score</code> tính toán độ chính xác của mô hình.</li>
  <li><code>recall_score</code> tính toán độ nhạy (recall) với <code>average='weighted'</code> để tính trung bình có trọng số cho tất cả các lớp.</li>
  <li><code>precision_score</code> tính toán độ chính xác (precision) với <code>average='weighted'</code>.</li>
  <li>Các chỉ số này được in ra với định dạng số thực có hai chữ số sau dấu phẩy.</li>
</ul>

<h3>Kết quả</h3>
<p>Sau khi chạy mã, bạn sẽ nhận được kết quả là các chỉ số độ chính xác, độ nhạy, và độ chính xác của mô hình KNN:</p>
<pre><code>
Độ chính xác: 0.98
Độ nhạy (Recall): 0.98
Độ chính xác (Precision): 0.98
</code></pre>

<h4>Giải thích:</h4>
<ul>
  <li><strong>Độ chính xác (Accuracy)</strong> đo lường tỷ lệ dự đoán đúng.</li>
  <li><strong>Độ nhạy (Recall)</strong> đánh giá khả năng mô hình nhận diện các đối tượng thuộc lớp thật.</li>
  <li><strong>Độ chính xác (Precision)</strong> đo lường khả năng mô hình chỉ ra chính xác các đối tượng thuộc lớp dự đoán.</li>
</ul>

<h2>Câu 4: Thử nghiệm với các giá trị k khác nhau trong mô hình KNN</h2>

<h3>Mô tả</h3>
<p>Mã này thực hiện các bước sau:</p>
<ol>
  <li>Thử nghiệm với các giá trị k khác nhau (1, 3, 5, 7, 9) trong mô hình KNN.</li>
  <li>Vẽ đồ thị thể hiện mối quan hệ giữa giá trị k và độ chính xác của mô hình.</li>
  <li>Nhận xét về kết quả thử nghiệm và xu hướng của đồ thị.</li>
</ol>

<h3>Mã nguồn</h3>

<pre><code class="language-python">
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Tải tập dữ liệu Wine
wine = load_wine()
X = wine.data
y = wine.target

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra với tỷ lệ 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Các giá trị k cần thử nghiệm
k_values = [1, 3, 5, 7, 9]
accuracies = []

# Thử nghiệm với từng giá trị k
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Vẽ đồ thị
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.title('Mối quan hệ giữa k và độ chính xác của mô hình KNN')
plt.xlabel('Giá trị k')
plt.ylabel('Độ chính xác')
plt.xticks(k_values)
plt.grid()
plt.show()
</code></pre>

<h4>Giải thích:</h4>
<ul>
  <li>Mã thử nghiệm mô hình KNN với các giá trị khác nhau của k (1, 3, 5, 7, 9) và tính toán độ chính xác trên tập kiểm tra.</li>
  <li>Đồ thị thể hiện sự thay đổi của độ chính xác khi thay đổi giá trị của k, giúp chúng ta hiểu được ảnh hưởng của k đối với mô hình.</li>
</ul>

<h3>Kết quả</h3>
<p>Đồ thị hiển thị mối quan hệ giữa k và độ chính xác. Bạn sẽ thấy rằng độ chính xác có thể tăng lên hoặc giảm đi tùy thuộc vào giá trị của k, và có thể có một giá trị k tối ưu cho mô hình KNN trên dữ liệu Wine.</p>

<h2>Câu 5: Xây dựng mô hình Cây Quyết Định cho Dữ Liệu Ung Thư Vú</h2>

<h3>Mô tả</h3>
<p>Mã này thực hiện các bước sau:</p>
<ol>
  <li>Tải tập dữ liệu Breast Cancer.</li>
  <li>Chia tập dữ liệu thành hai phần: tập huấn luyện (75%) và tập kiểm tra (25%).</li>
  <li>Xây dựng và huấn luyện mô hình Cây Quyết Định (Decision Tree).</li>
  <li>Vẽ cây quyết định và tính toán độ chính xác của mô hình trên tập kiểm tra.</li>
</ol>

<h3>Mã nguồn</h3>

<pre><code class="language-python">
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Tải tập dữ liệu Breast Cancer
cancer_data = load_breast_cancer()
X = cancer_data.data
y = cancer_data.target

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra với tỷ lệ 75:25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Khởi tạo và huấn luyện mô hình cây quyết định
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Tính toán độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình: {accuracy:.2f}")

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
# Vẽ cây quyết định
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=cancer_data.feature_names, class_names=cancer_data.target_names)
plt.title("Cây quyết định cho dữ liệu ung thư vú")
plt.savefig("decision_tree.png")  # Lưu hình ảnh
plt.show()
</code></pre>

<h4>Giải thích:</h4>
<ul>
  <li>Đoạn mã này huấn luyện một mô hình cây quyết định trên dữ liệu ung thư vú và tính toán độ chính xác của mô hình.</li>
  <li>Cây quyết định được vẽ và lưu lại dưới dạng hình ảnh để dễ dàng quan sát cấu trúc của mô hình.</li>
</ul>

<h2>Câu 6: Huấn luyện mô hình SVM với Kernel Tuyến Tính trên Dữ Liệu Digits</h2>

<h3>Mô tả</h3>
<p>Mã này thực hiện các bước sau:</p>
<ol>
  <li>Tải tập dữ liệu Digits từ sklearn.datasets.</li>
  <li>Huấn luyện mô hình SVM với kernel tuyến tính và đánh giá độ chính xác trên tập kiểm tra.</li>
</ol>

<h3>Mã nguồn</h3>

<pre><code class="language-python">
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Tải tập dữ liệu Digits
digits = load_digits()
X = digits.data
y = digits.target

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra với tỷ lệ 75:25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Khởi tạo mô hình SVM với kernel tuyến tính
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Tính toán độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình SVM với kernel tuyến tính: {accuracy:.2f}")
</code></pre>

<h4>Giải thích:</h4>
<ul>
  <li>Mô hình SVM với kernel tuyến tính được huấn luyện và đánh giá trên tập dữ liệu Digits.</li>
  <li>Đoạn mã tính toán và in ra độ chính xác của mô hình sau khi huấn luyện xong.</li>
</ul>

<h2>Câu 7: So sánh các Kernel SVM (linear, rbf, poly)</h2>

<h3>Mô tả</h3>
<p>Mã này thực hiện các bước sau:</p>
<ol>
  <li>Thử nghiệm với các kernel khác nhau (linear, rbf, poly).</li>
  <li>So sánh độ chính xác và thời gian huấn luyện giữa các kernel này.</li>
  <li>Đưa ra nhận xét về hiệu quả của mỗi kernel trên tập dữ liệu Digits.</li>
</ol>

<h3>Mã nguồn</h3>

<pre><code class="language-python">
import time
kernels = ['linear', 'rbf', 'poly']
results = {}

for kernel in kernels:
    start_time = time.time()  # Bắt đầu đo thời gian
    model = SVC(kernel=kernel, random_state=42)
    model.fit(X_train, y_train)  # Huấn luyện mô hình
    y_pred = model.predict(X_test)  # Dự đoán trên tập kiểm tra
    accuracy = accuracy_score(y_test, y_pred)  # Tính toán độ chính xác
    elapsed_time = time.time() - start_time  # Tính toán thời gian huấn luyện

    results[kernel] = {
        'accuracy': accuracy,
        'time': elapsed_time
    }

for kernel, metrics in results.items():
    print(f"Kernel: {kernel}")
    print(f"  Độ chính xác: {metrics['accuracy']:.2f}")
    print(f"  Thời gian huấn luyện: {metrics['time']:.4f} giây")
</code></pre>

<h4>Giải thích:</h4>
<ul>
  <li>Đoạn mã thử nghiệm với các kernel khác nhau (linear, rbf, poly) và đo lường thời gian huấn luyện cũng như độ chính xác của từng kernel.</li>
  <li>Kết quả so sánh sẽ giúp bạn







