import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load the cleaned dataset
df = pd.read_csv("cleaned_reviews_sampled.csv")

# Drop any rows where Cleaned_Text is NaN or empty
df = df.dropna(subset=['Cleaned_Text'])  # Remove NaN values
df = df[df['Cleaned_Text'].str.strip() != ""]  # Remove empty strings

# Initialize TF-IDF Vectorizer (limit to 10,000 features for efficiency)
vectorizer = TfidfVectorizer(max_features=10000)

# Convert text into numerical features
X = vectorizer.fit_transform(df['Cleaned_Text'])  # Feature matrix
y = df['Sentiment']  # Labels

# Save the TF-IDF model for later use
with open("tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("TF-IDF transformation completed!")
print("Feature Matrix Shape:", X.shape)  # Check dimensions of numerical features