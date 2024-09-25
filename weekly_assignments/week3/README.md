# Naive Bayes Classifier for Drug Prediction

This project implements a machine learning model using **Naive Bayes Classifier** to predict the type of drug a patient should take based on several health features.

## Technologies Used
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn (sklearn)**: For machine learning tasks such as model training, data splitting, and performance evaluation.
  - `train_test_split`: Splits the dataset into training and testing sets.
  - `LabelEncoder`: Encodes categorical variables (e.g., sex, blood pressure, cholesterol) into numerical values.
  - `GaussianNB`: Implements the Naive Bayes Classifier using Gaussian distribution.
  - `accuracy_score`: Computes the accuracy of the model.

## Algorithm Used: Naive Bayes Classifier

### Naive Bayes Classifier
The **Naive Bayes Classifier** is a probabilistic machine learning algorithm based on Bayes' Theorem, with the assumption of independence between features. This project specifically uses **Gaussian Naive Bayes**, which assumes that the data follows a Gaussian (normal) distribution.

Steps:
1. **Label Encoding**: Categorical variables such as 'Sex', 'BP', and 'Cholesterol' are transformed into numerical labels using `LabelEncoder`.
2. **Data Splitting**: The dataset is split into features (`X`: Age, Sex, BP, Cholesterol, Na_to_K) and the target variable (`y`: Drug). The data is then divided into training and test sets using `train_test_split`.
3. **Model Training**: A Gaussian Naive Bayes model is initialized and trained on the training set.
4. **Prediction and Evaluation**: The model predicts the target variable on the test set, and its accuracy is computed using `accuracy_score`.

### Code Breakdown
```python
#import library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Encode categorical variables
le_sex = LabelEncoder()
le_bp = LabelEncoder()
le_cholesterol = LabelEncoder()

df['Sex'] = le_sex.fit_transform(df['Sex'])
df['BP'] = le_bp.fit_transform(df['BP'])
df['Cholesterol'] = le_cholesterol.fit_transform(df['Cholesterol'])

# Split data into features (X) and target (y)
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = df['Drug']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Naive Bayes Classifier
nb_classifier = GaussianNB()

# Train the model
nb_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = nb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the results
print(f"Predictions: {y_pred}")
print(f"Accuracy: {accuracy}")
