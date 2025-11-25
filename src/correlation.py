import pandas as pd

def calculate_daily_returns(df: pd.DataFrame, price_col='close') -> pd.Series:
    """
    Compute daily stock returns as percentage change.

    Args:
        df (pd.DataFrame): Stock DataFrame.
        price_col (str): Column to calculate returns from.

    Returns:
        pd.Series: Daily percentage returns.
    """
    return df[price_col].pct_change()

def aggregate_daily_sentiment(df: pd.DataFrame, date_col='date', sentiment_col='sentiment') -> pd.Series:
    """
    Aggregate multiple sentiment scores per day by taking the mean.

    Args:
        df (pd.DataFrame): DataFrame with sentiment scores.
        date_col (str): Column with dates.
        sentiment_col (str): Column with sentiment scores.

    Returns:
        pd.Series: Average daily sentiment scores.
    """
    return df.groupby(date_col)[sentiment_col].mean()

def compute_correlation(sentiments: pd.Series, returns: pd.Series) -> float:
    """
    Compute Pearson correlation between daily sentiment and stock returns.

    Args:
        sentiments (pd.Series): Daily average sentiment scores.
        returns (pd.Series): Daily stock returns.

    Returns:
        float: Correlation coefficient.
    """
    return sentiments.corr(returns)
