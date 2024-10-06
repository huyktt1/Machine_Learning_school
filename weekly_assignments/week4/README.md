# Phân Loại Văn Bản Bằng KNN
## Giới Thiệu

Trong phần này, chúng ta sẽ sử dụng thuật toán KNN (K-Nearest Neighbors) để phân loại văn bản dựa trên dữ liệu đã cho. KNN là một phương pháp học máy đơn giản nhưng hiệu quả, thường được sử dụng cho các bài toán phân loại và hồi quy.

Đoạn mã dưới đây sẽ cài đặt các thư viện cần thiết cho dự án:

```python
import pandas as pd
import numpy as np
Giải Thích:
import pandas as pd: Nhập thư viện pandas, thư viện mạnh mẽ để xử lý và phân tích dữ liệu.
import numpy as np: Nhập thư viện numpy, thư viện hỗ trợ tính toán số học trong Python, đặc biệt là cho các mảng.
Tải Dữ Liệu
================================= Đoạn mã dưới đây định nghĩa một hàm để tải dữ liệu từ một tệp CSV:
```

```
# Load the dataset
def load_csv(file_path):
    return pd.read_csv(file_path)
Giải Thích:
Hàm load_csv(file_path): Hàm này nhận vào một tham số file_path, đại diện cho đường dẫn đến tệp CSV mà bạn muốn tải.
pd.read_csv(file_path): Đọc tệp CSV và trả về một DataFrame.
Làm Sạch Dữ Liệu
================================= Đoạn mã dưới đây định nghĩa một hàm để làm sạch dữ liệu văn bản:
```

# Function to clean the text data
```
def clean_text(text):
    return text.replace(',', '').replace('.', '').lower()
Giải Thích:
Hàm clean_text(text): Hàm này làm sạch chuỗi văn bản bằng cách loại bỏ dấu phẩy và dấu chấm, đồng thời chuyển đổi mọi ký tự thành chữ thường.
Chia Dữ Liệu
================================= Đoạn mã dưới đây định nghĩa một hàm để chia dữ liệu thành các tập huấn luyện và kiểm tra:
```

# Function to split data into training and testing sets
```
def split_train_test(data, test_size):
    mask = np.random.rand(len(data)) < (1 - test_size)
    X_train = data[mask]
    X_test = data[~mask]
    y_train = X_train['Label'].values  # Use .values for correct indexing
    y_test = X_test['Label'].values      # Use .values for correct indexing
    return X_train['Text'], y_train, X_test['Text'], y_test
```
Giải Thích:
# Hàm split_train_test(data, test_size): Hàm này chia dữ liệu thành tập huấn luyện và tập kiểm tra.
# mask: Tạo một mảng boolean để chỉ định xem mỗi mẫu có thuộc tập huấn luyện hay không.
# X_train và X_test: Các tập chứa văn bản từ dữ liệu huấn luyện và kiểm tra.
# y_train và y_test: Các tập chứa nhãn tương ứng cho dữ liệu huấn luyện và kiểm tra.
Tính Tần Suất Từ
================================= Đoạn mã dưới đây định nghĩa một hàm để tính tần suất của các từ trong dữ liệu huấn luyện:


# Function to get word frequency from the training data
```
def get_words_frequency(X_train):
    words = ' '.join(X_train).split()
    bags = set(words)
    word_count = {word: words.count(word) for word in bags}
    return word_count, bags
```
# Giải Thích:
Hàm get_words_frequency(X_train): Hàm này tính toán tần suất của các từ trong tập huấn luyện.
words: Kết hợp tất cả các văn bản trong X_train thành một chuỗi.
bags: Tạo một tập hợp các từ duy nhất.
word_count: Tạo một từ điển chứa số lần xuất hiện của mỗi từ.
Chuyển Đổi Dữ Liệu
================================= Đoạn mã dưới đây định nghĩa một hàm để chuyển đổi dữ liệu văn bản thành vector tần suất:


# Function to transform the text data into a frequency vector
```
def transform(X, bags):
    vectors = []
    for text in X:
        vector = [text.split().count(word) for word in bags]
        vectors.append(vector)
    return np.array(vectors)
```

# Giải Thích:
Hàm transform(X, bags): Chuyển đổi các văn bản thành các vector dựa trên tần suất xuất hiện của từ.
vectors: Danh sách chứa các vector tần suất cho từng văn bản.
Triển Khai KNN
================================= Đoạn mã dưới đây định nghĩa lớp KNN cho phân loại văn bản:


# KNN Implementation
```
class KNNText:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for test_point in X:
            distances = np.linalg.norm(self.X_train - test_point, axis=1)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            predictions.append(max(set(k_nearest_labels), key=k_nearest_labels.count))
        return np.array(predictions)
```
# Giải Thích:
Lớp KNNText: Triển khai thuật toán KNN cho dữ liệu văn bản.
__init__(self, k=3): Khởi tạo lớp với tham số k, số lượng láng giềng gần nhất.
fit(self, X, y): Lưu trữ dữ liệu huấn luyện.
predict(self, X): Dự đoán nhãn cho các điểm dữ liệu kiểm tra bằng cách tính khoảng cách tới các điểm huấn luyện và lấy nhãn của k láng giềng gần nhất.
Thực Hiện Dự Đoán
================================= Đoạn mã dưới đây thực hiện các bước chính để tải, làm sạch dữ liệu, chia tách, và thực hiện dự đoán:

# Load and clean the data
```
data = load_csv('/content/Education.csv')
data['Text'] = data['Text'].apply(clean_text)
```
# Split the data into train and test sets
```X_train, y_train, X_test, y_test = split_train_test(data, 0.25)
```
# Get word frequencies and bags
```words_train_fre, bags = get_words_frequency(X_train)```

# Transform the train and test data
```words_train_fre = transform(X_train, bags)
words_test_fre = transform(X_test, bags)
```
# Initialize and fit KNN
```knn = KNNText(k=3)
knn.fit(words_train_fre, y_train)
```
# Make predictions
```predictions = knn.predict(words_test_fre)```

# Create a DataFrame for predictions
```pred_df = pd.DataFrame(predictions, columns=['Predict'])
pred_df.index = range(1, len(pred_df) + 1)
```

# Prepare the actual labels DataFrame, ensuring the index aligns properly
```
y_test_df = pd.DataFrame(y_test, columns=['Actual'])
y_test_df.index = range(1, len(y_test_df) + 1)
```
# Concatenate predictions and actual labels
```
result = pd.concat([pred_df, y_test_df], axis=1)
```
# Display the result
```
print(result)
``` 
# Giải Thích:
Tải và làm sạch dữ liệu: Tải dữ liệu từ tệp CSV và làm sạch văn bản.
Chia dữ liệu: Chia dữ liệu thành tập huấn luyện và kiểm tra.
Tính tần suất từ và chuyển đổi: Tính tần suất từ và chuyển đổi dữ liệu thành các vector tần suất.
Khởi tạo và huấn luyện KNN: Khởi tạo mô hình KNN và huấn luyện nó với dữ liệu huấn luyện.
Dự đoán: Dự đoán nhãn cho dữ liệu kiểm tra.
Tạo DataFrame cho kết quả: Tạo một DataFrame chứa dự đoán và nhãn thực tế.
Hiển thị kết quả: In kết quả ra màn hình.
'''
Mô hình KNN thực hiện dự đoán trên tập kiểm tra và kết quả dự đoán được lưu trữ trong một DataFrame. Sau đó, DataFrame chứa nhãn thực tế cũng được tạo ra và hai DataFrame được nối lại để so sánh dự đoán và nhãn thực tế.
Kết Luận
================================= Mô hình KNN đã được xây dựng và áp dụng thành công để phân loại văn bản dựa trên dữ liệu đầu vào. Kết quả dự đoán có thể được kiểm tra bằng cách so sánh với nhãn thực tế, giúp đánh giá độ chính xác của mô hình.
