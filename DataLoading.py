import pandas as pd

# Load the dataset (replace with your file path if needed)
file_path = "Reviews.csv"  # If running locally
df = pd.read_csv(file_path)

# Keep only relevant columns
df = df[['Text', 'Score']]

# Drop missing values
df = df.dropna()

# Convert Score to Sentiment Labels
def convert_score_to_sentiment(score):
    if score >= 4:
        return "Positive"
    elif score == 3:
        return "Neutral"
    else:
        return "Negative"

df['Sentiment'] = df['Score'].apply(convert_score_to_sentiment)

# Drop the original Score column
df = df[['Text', 'Sentiment']]

# Display first few rows
print(df.head())

# Save the processed data for future use
df.to_csv("processed_reviews.csv", index=False)
print("Processed dataset saved as 'processed_reviews.csv'.")