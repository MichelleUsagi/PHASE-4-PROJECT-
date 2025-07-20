import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

class TextPreprocessingPipeline:
    def __init__(self, text_col='text'):
        self.text_col = text_col
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.token_pattern = r"[a-zA-Z]+(?:'[a-z]+)?"
        self.tokenizer = RegexpTokenizer(self.token_pattern)

    def expand_contractions(self, text: str) -> str:
        contractions = {
            "can't": "can not", "won't": "will not", "n't": " not",
            "'re": " are", "'s": " is", "'m": " am",
            "'ll": " will", "'ve": " have", "'d": " would"
        }
        for contr, full in contractions.items():
            text = re.sub(contr, full, text)
        return text

    def clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['clean_text'] = (
            df[self.text_col].str.lower()
               .apply(self.expand_contractions)
               .str.replace(r"[^\w\s']", "", regex=True)
               .str.replace(r"\s+", " ", regex=True)
               .str.strip()
        )
        df['clean_text'] = df['clean_text'].apply(
            lambda txt: ' '.join(w for w in txt.split() if w not in self.stop_words)
        )
        return df[df['clean_text'].str.len() > 0]

    def tokenize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['tokens'] = df['clean_text'].apply(
            lambda txt: [self.lemmatizer.lemmatize(tok)
                         for tok in self.tokenizer.tokenize(txt)]
        )
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.clean_text(df)
        df = self.tokenize(df)
        return df


def contains_emoji(text: str) -> bool:
    return any(ord(c) > 127 for c in text)


def flag_urls(df: pd.DataFrame) -> pd.DataFrame:
    df['has_url'] = df['text'].str.contains(r'http\S+|www\S+|https\S+', regex=True)
    return df


def flag_mentions(df: pd.DataFrame) -> pd.DataFrame:
    df['has_reddit_mention'] = df['text'].str.contains(r'\bu/\w+|\br/\w+', regex=True)
    return df
