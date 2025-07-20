from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

def truncate_text(df: pd.DataFrame, col='clean_text', max_len=500) -> pd.DataFrame:
    df = df.copy()
    df['text_trunc'] = df[col].str.slice(0, max_len)
    return df

def add_counts(df: pd.DataFrame, text_col='text_trunc') -> pd.DataFrame:
    df = df.copy()
    df['char_count'] = df[text_col].str.len()
    df['word_count'] = df[text_col].str.split().str.len()
    return df

def add_sentiment(df: pd.DataFrame, text_col='text_trunc') -> pd.DataFrame:
    df = df.copy()
    sid = SentimentIntensityAnalyzer()
    df['sentiment'] = df[text_col].apply(lambda txt: sid.polarity_scores(txt)['compound'])
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = truncate_text(df)
    df = add_counts(df)
    df = add_sentiment(df)
    return df
