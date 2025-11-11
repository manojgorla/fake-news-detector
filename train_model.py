import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the data
true_news = pd.read_csv("data/True.csv")
fake_news = pd.read_csv("data/Fake.csv")

# Add labels: 1 = True, 0 = Fake
true_news["label"] = 1
fake_news["label"] = 0

# Combine both datasets
data = pd.concat([true_news, fake_news], axis=0)

# Shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)

# Convert text to TF-IDF features
tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)
tfidf_train = tfidf.fit_transform(x_train)
tfidf_test = tfidf.transform(x_test)

# Train the model
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Make predictions
y_pred = pac.predict(tfidf_test)

# Evaluate
score = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {round(score * 100, 2)}%")

# Show confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import joblib

# Save the trained model and TF-IDF vectorizer
joblib.dump(pac, "model/fake_news_model.pkl")
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")

print("\n✅ Model and vectorizer saved successfully!")
