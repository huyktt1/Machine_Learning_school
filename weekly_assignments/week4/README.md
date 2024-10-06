# Phân Loại Văn Bản Bằng KNN
## Giới Thiệu

Trong phần này, chúng ta sẽ sử dụng thuật toán KNN (K-Nearest Neighbors) để phân loại văn bản dựa trên dữ liệu đã cho. KNN là một phương pháp học máy đơn giản nhưng hiệu quả, thường được sử dụng cho các bài toán phân loại và hồi quy.

## Cài Đặt Thư Viện

```
import pandas as pd
import numpy as np
pandas: Thư viện dùng để xử lý và phân tích dữ liệu, cung cấp cấu trúc dữ liệu như DataFrame để lưu trữ và thao tác với dữ liệu.
numpy: Thư viện hỗ trợ tính toán số học và ma trận, cho phép thực hiện các phép toán với tốc độ cao.
Tải Dữ Liệu
'''

'''

# Load the dataset
def load_csv(file_path):
    return pd.read_csv(file_path)
load_csv: Hàm này nhận đầu vào là đường dẫn đến file CSV và sử dụng pd.read_csv() để đọc dữ liệu từ file đó. Kết quả trả về là một DataFrame chứa dữ liệu.
Làm Sạch Dữ Liệu Văn Bản
'''

'''
# Function to clean the text data
def clean_text(text):
    return text.replace(',', '').replace('.', '').lower()
clean_text: Hàm này được sử dụng để làm sạch dữ liệu văn bản. Nó loại bỏ các dấu câu (,, .) và chuyển đổi tất cả các ký tự về dạng chữ thường để giảm thiểu độ phức tạp trong việc phân tích.
Chia Dữ Liệu Thành Tập Huấn Luyện và Kiểm Tra
'''
'''
# Function to split data into training and testing sets
def split_train_test(data, test_size):
    mask = np.random.rand(len(data)) < (1 - test_size)
    X_train = data[mask]
    X_test = data[~mask]
    y_train = X_train['Label'].values  # Use .values for correct indexing
    y_test = X_test['Label'].values      # Use .values for correct indexing
    return X_train['Text'], y_train, X_test['Text'], y_test
'''
split_train_test: Hàm này chia dữ liệu thành hai tập: một tập dùng để huấn luyện mô hình và một tập dùng để kiểm tra độ chính xác của mô hình. test_size xác định tỷ lệ phần trăm dữ liệu được sử dụng cho tập kiểm tra. Dữ liệu được chia ngẫu nhiên bằng cách tạo một mặt nạ (mask) sử dụng numpy.
Tính Tần Suất Từ Trong Tập Huấn Luyện


'''
# Function to get word frequency from the training data
def get_words_frequency(X_train):
    words = ' '.join(X_train).split()
    bags = set(words)
    word_count = {word: words.count(word) for word in bags}
    return word_count, bags
'''
get_words_frequency: Hàm này tính tần suất xuất hiện của các từ trong tập huấn luyện. Từ các văn bản trong X_train, hàm sẽ tạo ra một danh sách các từ và đếm số lần xuất hiện của mỗi từ. Kết quả trả về là một từ điển chứa các từ và số lượng tương ứng cùng với một tập hợp các từ (bags).
Chuyển Đổi Dữ Liệu Văn Bản Thành Vector Tần Suất
'''
# Function to transform the text data into a frequency vector
def transform(X, bags):
    vectors = []
    for text in X:
        vector = [text.split().count(word) for word in bags]
        vectors.append(vector)
    return np.array(vectors)
'''
transform: Hàm này chuyển đổi dữ liệu văn bản thành một vector tần suất, trong đó mỗi phần tử của vector đại diện cho số lần xuất hiện của các từ trong tập hợp từ (bags) cho mỗi văn bản.
Cài Đặt KNN


'''
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
'''
KNNText: Lớp này triển khai thuật toán KNN.
init: Khởi tạo đối tượng KNN với số lượng láng giềng k.
fit: Hàm này được sử dụng để lưu trữ dữ liệu huấn luyện.
predict: Hàm này tính toán khoảng cách từ điểm kiểm tra đến các điểm huấn luyện, xác định các nhãn của k láng giềng gần nhất và dự đoán nhãn cho điểm kiểm tra dựa trên tần suất xuất hiện của các nhãn trong k láng giềng.
Tải và Làm Sạch Dữ Liệu


'''
# Load and clean the data
data = load_csv('/content/Education.csv')
data['Text'] = data['Text'].apply(clean_text)
'''
Dữ liệu được tải từ file CSV và sau đó áp dụng hàm clean_text để làm sạch cột văn bản.
Chia Dữ Liệu Thành Tập Huấn Luyện và Kiểm Tra

# Split the data into train and test sets
X_train, y_train, X_test, y_test = split_train_test(data, 0.25)
Dữ liệu được chia thành tập huấn luyện và kiểm tra với 25% dữ liệu cho tập kiểm tra.
Tính Tần Suất Từ và Bao Gồm
'''
# Get word frequencies and bags
words_train_fre, bags = get_words_frequency(X_train)
'''
# Transform the train and test data
words_train_fre = transform(X_train, bags)
words_test_fre = transform(X_test, bags)
Hàm get_words_frequency được gọi để tính tần suất từ cho tập huấn luyện. Sau đó, cả tập huấn luyện và tập kiểm tra được chuyển đổi thành vector tần suất thông qua hàm transform.
Khởi Tạo và Huấn Luyện KNN
'''

'''
# Initialize and fit KNN
knn = KNNText(k=3)
knn.fit(words_train_fre, y_train)
'''
Một đối tượng KNN được khởi tạo và huấn luyện bằng cách sử dụng dữ liệu tần suất từ và nhãn tương ứng.
Dự Đoán


'''
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
'''
Mô hình KNN thực hiện dự đoán trên tập kiểm tra và kết quả dự đoán được lưu trữ trong một DataFrame. Sau đó, DataFrame chứa nhãn thực tế cũng được tạo ra và hai DataFrame được nối lại để so sánh dự đoán và nhãn thực tế.
Kết Luận
================================= Mô hình KNN đã được xây dựng và áp dụng thành công để phân loại văn bản dựa trên dữ liệu đầu vào. Kết quả dự đoán có thể được kiểm tra bằng cách so sánh với nhãn thực tế, giúp đánh giá độ chính xác của mô hình.
