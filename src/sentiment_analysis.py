from textblob import TextBlob
import pandas as pd
from typing import Any

class SentimentAnalyzer:
    """
    Class to perform sentiment analysis on financial news headlines.
    """

    def get_sentiment(self, text: str) -> float:
        """
        Compute sentiment polarity of a headline.
        Returns a value between -1 (negative) and +1 (positive).

        Args:
            text (str): The headline text.

        Returns:
            float: Sentiment polarity score.
        """
        # TextBlob.sentiment is a runtime property implemented as a
        # cached_property and some type-checkers (pyright/pylance) don't
        # recognize the namedtuple fields like `polarity`.
        # Cast to Any to silence the type checker while preserving
        # the correct runtime behavior.
        sentiment: Any = TextBlob(text).sentiment
        return float(sentiment.polarity)

def apply_sentiment(df: pd.DataFrame, text_column='headline') -> pd.Series:
    """
    Apply sentiment analysis to a DataFrame column.

    Args:
        df (pd.DataFrame): DataFrame containing text data.
        text_column (str): Column name containing the text.

    Returns:
        pd.Series: Sentiment scores for each row.
    """
    analyzer = SentimentAnalyzer()
    return df[text_column].apply(analyzer.get_sentiment)
