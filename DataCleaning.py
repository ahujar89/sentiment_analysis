import pandas as pd
import re

# Load dataset
df = pd.read_csv("processed_reviews.csv")
df = df.sample(200000, random_state=42)  

# Function to clean text (FAST method)
def fast_clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove punctuation and special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.strip()  # Remove extra spaces
    return text

# Apply cleaning (super fast)
df['Cleaned_Text'] = df['Text'].apply(fast_clean_text)

# Keep only necessary columns
df = df[['Cleaned_Text', 'Sentiment']]

# Save cleaned data
df.to_csv("cleaned_reviews_sampled.csv", index=False)
print("Sampled & cleaned dataset saved as 'cleaned_reviews_sampled.csv'.")