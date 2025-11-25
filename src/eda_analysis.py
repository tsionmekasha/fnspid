# src/eda_analysis.py
from collections import Counter
import re
import pandas as pd

def clean_text(text):
    """
    Clean text by removing non-alphanumeric characters and converting to lowercase.
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()

def extract_keywords(df: pd.DataFrame, column='headline', top_n=20):
    """
    Extract the top N most common keywords from a text column.

    Args:
        df (pd.DataFrame): DataFrame with a text column.
        column (str): Column containing text.
        top_n (int): Number of top keywords to return.

    Returns:
        pd.DataFrame: Top N keywords with counts.
    """
    all_words = []
    for text in df[column]:
        clean = clean_text(str(text))
        all_words.extend(clean.split())
    
    word_counts = Counter(all_words)
    # Remove common stopwords (optional)
    stopwords = set([
        'the', 'and', 'for', 'to', 'of', 'in', 'on', 'with', 'a', 'an', 'is', 
        'as', 'by', 'at', 'from', 'after', 'will'
    ])
    for word in stopwords:
        if word in word_counts:
            del word_counts[word]
    
    top_keywords = word_counts.most_common(top_n)
    return pd.DataFrame(top_keywords, columns=['keyword', 'count'])
