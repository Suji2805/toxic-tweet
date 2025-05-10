# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("C:/Users/psuji/Downloads/toxic tweet classification/FinalBalancedDataset.csv")  # Replace with your dataset path

# Feature (text) and Target (label)
X = df['tweet']  # Column with tweet text
y = df['Toxicity']  # Column with 0 (non-toxic) and 1 (toxic)

# Text preprocessing: Convert text to TF-IDF features
tfidf = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features
X_tfidf = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Predictions
y_pred = dt.predict(X_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
