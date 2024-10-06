# Phân Loại Văn Bản Bằng KNN
=================================

## Giới Thiệu
=================================
Trong phần này, chúng ta sẽ sử dụng thuật toán KNN để phân loại văn bản dựa trên dữ liệu đã cho.

## Cài Đặt Thư Viện
=================================
```python
import pandas as pd
import numpy as np
Tải Dữ Liệu
=================================

python
Sao chép mã
# Load the dataset
def load_csv(file_path):
    return pd.read_csv(file_path)
Làm Sạch Dữ Liệu Văn Bản
=================================

python
Sao chép mã
# Function to clean the text data
def clean_text(text):
    return text.replace(',', '').replace('.', '').lower()
Chia Dữ Liệu Thành Tập Huấn Luyện và Kiểm Tra
=================================

python
Sao chép mã
# Function to split data into training and testing sets
def split_train_test(data, test_size):
    mask = np.random.rand(len(data)) < (1 - test_size)
    X_train = data[mask]
    X_test = data[~mask]
    y_train = X_train['Label'].values  # Use .values for correct indexing
    y_test = X_test['Label'].values      # Use .values for correct indexing
    return X_train['Text'], y_train, X_test['Text'], y_test
Tính Tần Suất Từ Trong Tập Huấn Luyện
=================================

python
Sao chép mã
# Function to get word frequency from the training data
def get_words_frequency(X_train):
    words = ' '.join(X_train).split()
    bags = set(words)
    word_count = {word: words.count(word) for word in bags}
    return word_count, bags
Chuyển Đổi Dữ Liệu Văn Bản Thành Vector Tần Suất
=================================

python
Sao chép mã
# Function to transform the text data into a frequency vector
def transform(X, bags):
    vectors = []
    for text in X:
        vector = [text.split().count(word) for word in bags]
        vectors.append(vector)
    return np.array(vectors)
Cài Đặt KNN
=================================

python
Sao chép mã
# KNN Implementation
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
Tải và Làm Sạch Dữ Liệu
=================================

python
Sao chép mã
# Load and clean the data
data = load_csv('/content/Education.csv')
data['Text'] = data['Text'].apply(clean_text)
Chia Dữ Liệu Thành Tập Huấn Luyện và Kiểm Tra
=================================

python
Sao chép mã
# Split the data into train and test sets
X_train, y_train, X_test, y_test = split_train_test(data, 0.25)
Tính Tần Suất Từ và Bao Gồm
=================================

python
Sao chép mã
# Get word frequencies and bags
words_train_fre, bags = get_words_frequency(X_train)

# Transform the train and test data
words_train_fre = transform(X_train, bags)
words_test_fre = transform(X_test, bags)
Khởi Tạo và Huấn Luyện KNN
=================================

python
Sao chép mã
# Initialize and fit KNN
knn = KNNText(k=3)
knn.fit(words_train_fre, y_train)
Dự Đoán
=================================

python
Sao chép mã
# Make predictions
predictions = knn.predict(words_test_fre)

# Create a DataFrame for predictions
pred_df = pd.DataFrame(predictions, columns=['Predict'])
pred_df.index = range(1, len(pred_df) + 1)

# Prepare the actual labels DataFrame, ensuring the index aligns properly
y_test_df = pd.DataFrame(y_test, columns=['Actual'])
y_test_df.index = range(1, len(y_test_df) + 1)

# Concatenate predictions and actual labels
result = pd.concat([pred_df, y_test_df], axis=1)

# Display the result
print(result)
Kết Luận
================================= Mô hình KNN đã được xây dựng và áp dụng thành công để phân loại văn bản dựa trên dữ liệu đầu vào.
