# Mô hình Naive Bayes Dự đoán Loại Thuốc (LAB1)

Dự án này triển khai một mô hình học máy sử dụng **Naive Bayes Classifier** để dự đoán loại thuốc mà bệnh nhân nên sử dụng dựa trên một số đặc điểm sức khỏe.

## Công nghệ Sử dụng
- **Pandas**: Dùng để xử lý và tiền xử lý dữ liệu.
- **Scikit-learn (sklearn)**: Dùng cho các tác vụ học máy như huấn luyện mô hình, chia dữ liệu, và đánh giá hiệu suất.
  - `train_test_split`: Chia bộ dữ liệu thành tập huấn luyện và kiểm tra.
  - `LabelEncoder`: Mã hóa các biến phân loại (giới tính, huyết áp, cholesterol) thành các giá trị số.
  - `GaussianNB`: Triển khai mô hình Naive Bayes với phân phối Gaussian.
  - `accuracy_score`: Tính toán độ chính xác của mô hình.

## Thuật toán Sử dụng: Naive Bayes Classifier

### Naive Bayes Classifier
**Naive Bayes Classifier** là một thuật toán học máy theo xác suất dựa trên Định lý Bayes, với giả định rằng các đặc trưng (features) độc lập với nhau. Dự án này sử dụng cụ thể mô hình **Gaussian Naive Bayes**, giả định rằng dữ liệu tuân theo phân phối Gaussian (chuẩn).

Các bước:
1. **Mã hóa nhãn**: Các biến phân loại như 'Sex' (Giới tính), 'BP' (Huyết áp), và 'Cholesterol' được chuyển đổi thành các nhãn số bằng `LabelEncoder`.
2. **Chia dữ liệu**: Bộ dữ liệu được chia thành các đặc trưng (`X`: Tuổi, Giới tính, Huyết áp, Cholesterol, Na_to_K) và biến mục tiêu (`y`: Loại thuốc). Sau đó, dữ liệu được chia thành tập huấn luyện và kiểm tra sử dụng `train_test_split`.
3. **Huấn luyện mô hình**: Mô hình Gaussian Naive Bayes được khởi tạo và huấn luyện trên tập huấn luyện.
4. **Dự đoán và đánh giá**: Mô hình dự đoán biến mục tiêu trên tập kiểm tra và tính toán độ chính xác bằng cách sử dụng `accuracy_score`.

### Chi tiết mã nguồn
```python
#import thư viện
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Mã hóa các biến phân loại
le_sex = LabelEncoder()
le_bp = LabelEncoder()
le_cholesterol = LabelEncoder()

df['Sex'] = le_sex.fit_transform(df['Sex'])
df['BP'] = le_bp.fit_transform(df['BP'])
df['Cholesterol'] = le_cholesterol.fit_transform(df['Cholesterol'])

# Chia dữ liệu thành đặc trưng (X) và mục tiêu (y)
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = df['Drug']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Khởi tạo mô hình Naive Bayes
nb_classifier = GaussianNB()

# Huấn luyện mô hình
nb_classifier.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = nb_classifier.predict(X_test)

# Tính độ chính xác
accuracy = accuracy_score(y_test, y_pred)

# In kết quả
print(f"Dự đoán: {y_pred}")
print(f"Độ chính xác: {accuracy}")


# Mô hình Naive Bayes Phân loại Văn bản (LAB2)

Dự án này triển khai một mô hình phân loại văn bản sử dụng **Naive Bayes Classifier** (MultinomialNB) để dự đoán nhãn của các văn bản giáo dục.

## Công nghệ Sử dụng
- **Pandas**: Dùng để xử lý và quản lý dữ liệu từ file CSV.
- **Scikit-learn (sklearn)**: Dùng cho các tác vụ học máy như phân loại văn bản, huấn luyện mô hình, và đánh giá hiệu suất.
  - `CountVectorizer`: Chuyển đổi văn bản thành các đặc trưng số (Bag of Words) để sử dụng trong mô hình.
  - `train_test_split`: Chia bộ dữ liệu thành tập huấn luyện và kiểm tra.
  - `MultinomialNB`: Thuật toán Naive Bayes dành cho bài toán phân loại văn bản.
  - `accuracy_score`: Tính toán độ chính xác của mô hình.

## Thuật toán Sử dụng: Naive Bayes Classifier

### Multinomial Naive Bayes
**Multinomial Naive Bayes** là một phiên bản của thuật toán Naive Bayes thường được sử dụng cho các bài toán phân loại văn bản, đặc biệt với dữ liệu dạng túi từ (Bag of Words). Nó giả định rằng tần số xuất hiện của từ trong văn bản tuân theo phân phối đa thức (Multinomial Distribution).

Các bước chính:
1. **Đọc dữ liệu**: Sử dụng `pandas` để đọc dữ liệu văn bản từ file CSV.
2. **Chuyển đổi văn bản thành đặc trưng số**: Sử dụng `CountVectorizer` để chuyển đổi văn bản thành các giá trị số đại diện cho tần số xuất hiện của từ.
3. **Chia dữ liệu**: Chia dữ liệu thành tập huấn luyện và kiểm tra với tỷ lệ 70:30.
4. **Huấn luyện mô hình**: Sử dụng `MultinomialNB` để huấn luyện mô hình trên dữ liệu huấn luyện.
5. **Dự đoán và đánh giá**: Mô hình sẽ dự đoán nhãn của tập kiểm tra và độ chính xác được tính bằng `accuracy_score`.
6. **Dự đoán trên văn bản mới**: Mô hình có thể dự đoán nhãn của văn bản mới được cung cấp.

### Chi tiết mã nguồn
```python
#import thư viện
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Bước 1: Đọc dữ liệu từ file CSV
df = pd.read_csv('/content/Education.csv')
df

# Bước 2: Chuyển đổi dữ liệu văn bản thành các đặc trưng số sử dụng CountVectorizer (Bag of Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Text'])

# Bước 3: Xác định biến mục tiêu (Nhãn)
y = df['Label']

# Bước 4: Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Bước 5: Khởi tạo mô hình Naive Bayes (MultinomialNB thường dùng cho phân loại văn bản)
nb_classifier = MultinomialNB()

# Bước 6: Huấn luyện mô hình
nb_classifier.fit(X_train, y_train)

# Bước 7: Dự đoán trên tập kiểm tra
y_pred = nb_classifier.predict(X_test)

# Bước 8: Tính toán độ chính xác của mô hình
accuracy = accuracy_score(y_test, y_pred)

# Bước 9: Dự đoán trên văn bản mới
new_text = ['The impact of educational reforms remains uncertain despite extensive research.']
new_text_transformed = vectorizer.transform(new_text)
prediction = nb_classifier.predict(new_text_transformed)

# In kết quả
print(f"Dự đoán: {y_pred}")
print(f"Độ chính xác: {accuracy}")
print(f"Dự đoán cho văn bản mới: {prediction[0]}")

# Mô hình Naive Bayes Phân loại Văn bản (LAB2)

Dự án này triển khai một mô hình phân loại văn bản sử dụng **Naive Bayes Classifier** (MultinomialNB) để dự đoán nhãn của các văn bản giáo dục.

## Công nghệ Sử dụng
- **Pandas**: Dùng để xử lý và quản lý dữ liệu từ file CSV.
- **Scikit-learn (sklearn)**: Dùng cho các tác vụ học máy như phân loại văn bản, huấn luyện mô hình, và đánh giá hiệu suất.
  - `CountVectorizer`: Chuyển đổi văn bản thành các đặc trưng số (Bag of Words) để sử dụng trong mô hình.
  - `train_test_split`: Chia bộ dữ liệu thành tập huấn luyện và kiểm tra.
  - `MultinomialNB`: Thuật toán Naive Bayes dành cho bài toán phân loại văn bản.
  - `accuracy_score`: Tính toán độ chính xác của mô hình.

## Thuật toán Sử dụng: Naive Bayes Classifier

### Multinomial Naive Bayes
**Multinomial Naive Bayes** là một phiên bản của thuật toán Naive Bayes thường được sử dụng cho các bài toán phân loại văn bản, đặc biệt với dữ liệu dạng túi từ (Bag of Words). Nó giả định rằng tần số xuất hiện của từ trong văn bản tuân theo phân phối đa thức (Multinomial Distribution).

Các bước chính:
1. **Đọc dữ liệu**: Sử dụng `pandas` để đọc dữ liệu văn bản từ file CSV.
2. **Chuyển đổi văn bản thành đặc trưng số**: Sử dụng `CountVectorizer` để chuyển đổi văn bản thành các giá trị số đại diện cho tần số xuất hiện của từ.
3. **Chia dữ liệu**: Chia dữ liệu thành tập huấn luyện và kiểm tra với tỷ lệ 70:30.
4. **Huấn luyện mô hình**: Sử dụng `MultinomialNB` để huấn luyện mô hình trên dữ liệu huấn luyện.
5. **Dự đoán và đánh giá**: Mô hình sẽ dự đoán nhãn của tập kiểm tra và độ chính xác được tính bằng `accuracy_score`.
6. **Dự đoán trên văn bản mới**: Mô hình có thể dự đoán nhãn của văn bản mới được cung cấp.

### Chi tiết mã nguồn
```python
#import thư viện
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Bước 1: Đọc dữ liệu từ file CSV
df = pd.read_csv('/content/Education.csv')
df

# Bước 2: Chuyển đổi dữ liệu văn bản thành các đặc trưng số sử dụng CountVectorizer (Bag of Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Text'])

# Bước 3: Xác định biến mục tiêu (Nhãn)
y = df['Label']

# Bước 4: Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Bước 5: Khởi tạo mô hình Naive Bayes (MultinomialNB thường dùng cho phân loại văn bản)
nb_classifier = MultinomialNB()

# Bước 6: Huấn luyện mô hình
nb_classifier.fit(X_train, y_train)

# Bước 7: Dự đoán trên tập kiểm tra
y_pred = nb_classifier.predict(X_test)

# Bước 8: Tính toán độ chính xác của mô hình
accuracy = accuracy_score(y_test, y_pred)

# Bước 9: Dự đoán trên văn bản mới
new_text = ['The impact of educational reforms remains uncertain despite extensive research.']
new_text_transformed = vectorizer.transform(new_text)
prediction = nb_classifier.predict(new_text_transformed)

# In kết quả
print(f"Dự đoán: {y_pred}")
print(f"Độ chính xác: {accuracy}")
print(f"Dự đoán cho văn bản mới: {prediction[0]}")
