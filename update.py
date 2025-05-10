# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Streamlit page config
st.set_page_config(page_title="Toxic Tweet Classifier", layout="wide")
st.title("\U0001F4AC Toxic Tweet Classification App")

# Upload Dataset
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Sample of Uploaded Dataset")
    st.write(df.head())

    # Feature and Target
    X = df['tweet']
    y = df['Toxicity']

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Model Training
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)

    # Prediction
    y_pred = dt.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    st.metric("\U0001F4CA Model Accuracy", f"{accuracy*100:.2f}%")

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("Confusion Matrix Heatmap")
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Toxic', 'Toxic'], yticklabels=['Non-Toxic', 'Toxic'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)

    # Feature Importance Visualization
    st.subheader("Feature Importances")
    feature_importances = pd.DataFrame({
        'feature': tfidf.get_feature_names_out(),
        'importance': dt.feature_importances_
    })

    top_features = feature_importances.sort_values(by='importance', ascending=False).head(20)

    fig2 = px.bar(top_features[::-1], x='importance', y='feature', orientation='h',
                  labels={'importance':'Importance', 'feature':'Top Features'},
                  title='Top 20 Important Words')
    st.plotly_chart(fig2)

    # User Input Prediction
    st.subheader("Test on Your Own Tweet")
    user_input = st.text_area("Enter a tweet to classify:")

    if st.button("Classify Tweet"):
        user_input_vector = tfidf.transform([user_input])
        prediction = dt.predict(user_input_vector)
        label = "\U0001F7E2 Non-Toxic" if prediction[0] == 0 else "\U0001F534 Toxic"
        st.success(f"Prediction: {label}")

else:
    st.info("Awaiting for CSV file to be uploaded.")
    st.stop()

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit | Toxic Tweet Classifier")
