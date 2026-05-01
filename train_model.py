import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import joblib
import os

def train():
    print("Loading Zendesk CSV Data...")
    csv_file = "zendesk_tickets_export_20260501_090018.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: Could not find {csv_file}")
        return
        
    df = pd.read_csv(csv_file)
    
    # We need to extract the text and the label
    # Text = Subject + Public Comments
    # Label = Main Category
    
    print(f"Found {len(df)} total tickets.")
    
    # Drop rows without a Main Category
    df = df.dropna(subset=['Main Category'])
    print(f"Tickets with a Main Category to train on: {len(df)}")
    
    # Combine text
    df['Subject'] = df['Subject'].fillna('')
    df['Public Comments'] = df['Public Comments'].fillna('')
    df['text'] = df['Subject'] + " " + df['Public Comments']
    
    X = df['text']
    y = df['Main Category']
    
    print("Training the Machine Learning model (TF-IDF + LinearSVC)...")
    # Create an ML Pipeline:
    # 1. TfidfVectorizer converts text into a math matrix
    # 2. LinearSVC is a fast, highly accurate classifier for text
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.9, min_df=2)),
        ('clf', LinearSVC(random_state=42, dual=False))
    ])
    
    pipeline.fit(X, y)
    
    model_path = "category_model.pkl"
    joblib.dump(pipeline, model_path)
    
    print(f"Model successfully trained and saved to {model_path}!")
    
if __name__ == "__main__":
    train()
