import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the cleaned dataset
df = pd.read_csv("cleaned_reviews_sampled.csv")

# Drop any missing values again just to be safe
df = df.dropna(subset=['Cleaned_Text'])
df = df[df['Cleaned_Text'].str.strip() != ""]

# Load the saved TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Transform the text data into numerical vectors
X = vectorizer.transform(df['Cleaned_Text'])  # Convert text into feature matrix
y = df['Sentiment']  # Labels

# Split the dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the trained model
with open("sentiment_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Display detailed classification report
print(classification_report(y_test, y_pred))