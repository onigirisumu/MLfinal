#!/usr/bin/env python3
"""
Movie Recommendation Model Training Script
Run with: python train_model.py
"""

import pandas as pd
import ast
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report

def load_data(filepath="tmdb_5000_movies.csv"):
    """Load and preprocess the movie data"""
    print("🔍 Loading data...")
    df = pd.read_csv(filepath)
    
    def parse_genres(genre_str):
        try:
            genres = ast.literal_eval(genre_str)
            return [g['name'] for g in genres]
        except (ValueError, SyntaxError):
            return []
    
    df['genres'] = df['genres'].apply(parse_genres)
    df = df[['overview', 'genres']].dropna()
    return df[df['genres'].map(len) > 0]

def train_and_save():
    """Main training pipeline"""
    # 1. Load Data
    df = load_data()
    print(f"✅ Loaded {len(df)} movies")

    # 2. Prepare Targets
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['genres'])
    print(f"🎭 {len(mlb.classes_)} genres detected")

    # 3. Create Embeddings
    print("🔮 Generating text embeddings...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    X = embedding_model.encode(df['overview'].tolist(), show_progress_bar=True)

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Train Model
    print("🏋️ Training classifier...")
    classifier = OneVsRestClassifier(
        LogisticRegression(max_iter=500, random_state=42)
    )
    classifier.fit(X_train, y_train)

    # 6. Evaluate
    print("\n📊 Evaluation Results:")
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=mlb.classes_))

    # 7. Save Models
    print("\n💾 Saving models...")
    joblib.dump(classifier, 'multi_label_model.pkl')
    joblib.dump(mlb, 'multi_label_binarizer.pkl')
    joblib.dump(df[['title', 'overview', 'genres']], 'movies_data.pkl') 
    print("🎉 All models saved successfully!")

if __name__ == "__main__":
    train_and_save()