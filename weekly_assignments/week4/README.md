K-Nearest Neighbors (KNN) Text Classification
This project implements a K-Nearest Neighbors (KNN) algorithm to classify text data using frequency vectors derived from the training dataset. Below is a breakdown of the key components of the code.

1. Import Libraries
python
Sao chép mã
import pandas as pd
import numpy as np
We use pandas for data manipulation and numpy for numerical operations.

2. Load the Dataset
python
Sao chép mã
def load_csv(file_path):
    return pd.read_csv(file_path)
The load_csv function takes a file path as input and returns a DataFrame containing the loaded dataset. This function uses pd.read_csv() to read a CSV file into a DataFrame.

3. Clean Text Data
python
Sao chép mã
def clean_text(text):
    return text.replace(',', '').replace('.', '').lower()
The clean_text function cleans the text by removing commas and periods and converting all characters to lowercase. This is an important step in text processing to ensure uniformity.

4. Split Data into Training and Testing Sets
python
Sao chép mã
def split_train_test(data, test_size):
    mask = np.random.rand(len(data)) < (1 - test_size)
    X_train = data[mask]
    X_test = data[~mask]
    y_train = X_train['Label'].values  # Use .values for correct indexing
    y_test = X_test['Label'].values      # Use .values for correct indexing
    return X_train['Text'], y_train, X_test['Text'], y_test
The split_train_test function divides the dataset into training and testing sets based on the specified test_size. A random mask is created to select the training samples, while the remaining samples are used for testing. The function returns the text and labels for both sets.

5. Get Word Frequency
python
Sao chép mã
def get_words_frequency(X_train):
    words = ' '.join(X_train).split()
    bags = set(words)
    word_count = {word: words.count(word) for word in bags}
    return word_count, bags
The get_words_frequency function generates a frequency count of the words from the training set. It joins all text samples, splits them into words, and counts the occurrence of each unique word, returning both the word count and a set of unique words (bags).

6. Transform Text Data into Frequency Vectors
python
Sao chép mã
def transform(X, bags):
    vectors = []
    for text in X:
        vector = [text.split().count(word) for word in bags]
        vectors.append(vector)
    return np.array(vectors)
The transform function converts the text data into frequency vectors based on the unique words obtained from the training data. Each text sample is represented as a vector of word counts corresponding to the unique words (bags).

7. KNN Implementation
python
Sao chép mã
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
The KNNText class implements the KNN algorithm. The constructor initializes the number of neighbors (k) and placeholders for training data. The fit method stores the training data, and the predict method calculates the distances from the test points to the training points, retrieves the nearest neighbors, and predicts the label based on majority voting.

8. Main Code Execution
python
Sao chép mã
# Load and clean the data
data = load_csv('/content/Education.csv')
data['Text'] = data['Text'].apply(clean_text)

# Split the data into train and test sets
X_train, y_train, X_test, y_test = split_train_test(data, 0.25)

# Get word frequencies and bags
words_train_fre, bags = get_words_frequency(X_train)

# Transform the train and test data
words_train_fre = transform(X_train, bags)
words_test_fre = transform(X_test, bags)

# Initialize and fit KNN
knn = KNNText(k=3)
knn.fit(words_train_fre, y_train)

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
The dataset is loaded and cleaned using the load_csv and clean_text functions.
The data is split into training and testing sets using the split_train_test function.
Word frequencies are obtained and transformed into frequency vectors for both training and testing data.
An instance of the KNNText class is created, and the model is trained using the training data.
Predictions are made on the test data, and a DataFrame is created to display both the predicted and actual labels.
Conclusion
This implementation provides a foundational approach to text classification using KNN, leveraging frequency-based feature representation. You can modify the k parameter in the KNNText class to explore its impact on classification performance.


