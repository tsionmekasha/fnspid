# src/pipelines.py

import pandas as pd
from src.data_loader import load_news_data, load_stock_data
from src.eda_analysis import extract_keywords
from src.sentiment_analysis import apply_sentiment
from src.correlation import aggregate_daily_sentiment, compute_correlation
from src.technical_indicators import moving_average, rsi, macd
from src.pynance_utils import calculate_daily_returns_pynance, calculate_volatility

def run_eda_pipeline(news_csv: str):
    """
    End-to-end EDA pipeline:
        1. Load news data
        2. Headline length, publisher counts
        3. Keyword extraction
        4. Sentiment analysis

    Returns:
        news_df (pd.DataFrame), daily_sentiment (pd.Series), top_keywords (pd.DataFrame)
    """
    news_df = load_news_data(news_csv)
    news_df['headline_length'] = news_df['headline'].apply(len)

    # Publisher counts
    publisher_counts = news_df['publisher'].value_counts()
    print("Top publishers:\n", publisher_counts.head(10))

    # Keywords
    top_keywords = extract_keywords(news_df, top_n=20)
    print("Top 20 keywords:\n", top_keywords)

    # Sentiment
    news_df['sentiment'] = apply_sentiment(news_df, 'headline')
    daily_sentiment = aggregate_daily_sentiment(news_df)

    return news_df, daily_sentiment, top_keywords

def run_correlation_pipeline(stock_csv: str, daily_sentiment: pd.Series):
    """
    End-to-end correlation pipeline:
        1. Load stock data
        2. Calculate PyNance daily returns and volatility
        3. Calculate TA-Lib indicators (SMA, RSI, MACD)
        4. Align with daily sentiment
        5. Compute correlation

    Returns:
        stock_df (pd.DataFrame), daily_returns (pd.Series), volatility (pd.Series), correlation (float)
    """
    stock_df = load_stock_data(stock_csv)

    # PyNance metrics
    daily_returns = calculate_daily_returns_pynance(stock_df)
    volatility = calculate_volatility(stock_df)

    # TA-Lib indicators
    # moving_average uses the `timeperiod` kwarg to align with TA-Lib API
    stock_df['SMA_7'] = moving_average(stock_df['close'], timeperiod=7)
    stock_df['RSI_14'] = rsi(stock_df['close'], timeperiod=14)
    stock_df['MACD'], stock_df['MACD_signal'], stock_df['MACD_hist'] = macd(stock_df['close'])

    # Align sentiment with stock dates
    aligned_returns = daily_returns[daily_sentiment.index.min():daily_sentiment.index.max()]

    # Correlation
    correlation = compute_correlation(daily_sentiment, aligned_returns)
    print(f"Correlation between sentiment & stock returns: {correlation:.3f}")

    return stock_df, daily_returns, volatility, correlation
