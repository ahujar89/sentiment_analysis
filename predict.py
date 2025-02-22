import pickle

# Load the trained model and vectorizer
with open("sentiment_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Function to predict sentiment
def predict_sentiment(review):
    review_vectorized = vectorizer.transform([review])  # Convert to TF-IDF vector
    prediction = model.predict(review_vectorized)  # Get prediction
    return prediction[0]

# Predefined test reviews
test_reviews = [
    "This product is amazing! I love it so much!",
    "Worst purchase ever. Waste of money!",
    "The quality is okay, but not the best.",
    "Absolutely fantastic, would buy again!",
    "Horrible experience, would not recommend."
]

# Run predictions on test examples
print("\n=== Predicted Sentiments for Sample Reviews ===\n")
for review in test_reviews:
    sentiment = predict_sentiment(review)
    print(f"Review: {review} \nPredicted Sentiment: {sentiment}\n")

# Real-time user input loop
print("\n=== Enter Your Own Reviews for Sentiment Prediction ===")
print("Type 'exit' to quit.\n")

while True:
    user_review = input("Enter a review: ")
    if user_review.lower() == 'exit':
        print("Exiting program. Goodbye!")
        break
    sentiment = predict_sentiment(user_review)
    print(f"Predicted Sentiment: {sentiment}\n")