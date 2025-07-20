import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.preprocessing import TextPreprocessingPipeline, contains_emoji, flag_urls, flag_mentions
from src.feature_engineering import build_features

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.drop_duplicates(subset='text', inplace=True)
    return df

def main():
    # 1) Load raw data
    raw_path = os.path.join(os.getcwd(), 'data', 'raw.csv')
    df = load_data(raw_path)

    # 2) Flag extras
    df = flag_urls(df)
    df = flag_mentions(df)
    df['has_emoji'] = df['text'].apply(contains_emoji)

    # 3) Preprocess & feature‐engineer
    df = TextPreprocessingPipeline(text_col='text').run(df)
    df = build_features(df)

    # 4) Encode and split
    df['label'] = df['class'].map({'non-suicide': 0, 'suicide': 1})
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'],
        test_size=0.3, random_state=42, stratify=df['label']
    )

    # 5) Train model pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    # 6) Serialize
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/logreg_pipeline.pkl')
    print("Saved model to models/logreg_pipeline.pkl")

if __name__ == "__main__":
    main()
