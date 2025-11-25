# main.py
import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import load_news_data, load_stock_data
from src.sentiment_analysis import apply_sentiment
from src.technical_indicators import moving_average, rsi, macd
from src.correlation import calculate_daily_returns, aggregate_daily_sentiment, compute_correlation
from src.eda_analysis import extract_keywords  # <-- new section for Task 1 enhancement

# -------------------------------
# Step 1: Load datasets
# -------------------------------
news_df = load_news_data('data/news.csv')
stock_df = load_stock_data('data/stock.csv')

# -------------------------------
# Step 2: Task 1 - EDA
# -------------------------------

# Headline length analysis
news_df['headline_length'] = news_df['headline'].apply(len)
print("Headline length stats:")
print(news_df['headline_length'].describe())

# Publisher counts
publisher_counts = news_df['publisher'].value_counts()
print("\nTop publishers by number of articles:")
print(publisher_counts.head(10))

# Plot headline length distribution
plt.figure(figsize=(8,5))
plt.hist(news_df['headline_length'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Headline Lengths')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.savefig('headline_length_distribution.png')
plt.close()

# Plot top 10 publishers
plt.figure(figsize=(10,5))
publisher_counts.head(10).plot(kind='bar', color='orange')
plt.title('Top 10 Publishers by Article Count')
plt.ylabel('Number of Articles')
plt.xlabel('Publisher')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('top_publishers.png')
plt.close()

# -------------------------------
# Step 2a: Keyword / Topic Extraction (New)
# -------------------------------
top_keywords = extract_keywords(news_df, top_n=20)
print("\nTop 20 keywords in headlines:")
print(top_keywords)

# Plot top keywords
plt.figure(figsize=(12,5))
plt.bar(top_keywords['keyword'], top_keywords['count'], color='skyblue')
plt.title('Top 20 Keywords in Financial News Headlines')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('top_keywords.png')
plt.close()

# -------------------------------
# Step 3: Task 1 - Sentiment Analysis
# -------------------------------
news_df['sentiment'] = apply_sentiment(news_df, 'headline')

# Aggregate daily sentiment
daily_sentiment = aggregate_daily_sentiment(news_df)

# Plot daily sentiment over time
plt.figure(figsize=(12,5))
daily_sentiment.plot(color='green')
plt.title('Daily Average News Sentiment')
plt.xlabel('Date')
plt.ylabel('Average Sentiment')
plt.tight_layout()
plt.savefig('daily_sentiment.png')
plt.close()

# -------------------------------
# Step 4: Task 2 - Technical Indicators
# -------------------------------
stock_df['SMA_7'] = moving_average(stock_df['close'], timeperiod=7)
stock_df['RSI_14'] = rsi(stock_df['close'], timeperiod=14)
stock_df['MACD'], stock_df['MACD_signal'], stock_df['MACD_hist'] = macd(stock_df['close'])

# Plot stock close price and SMA
plt.figure(figsize=(12,5))
plt.plot(stock_df['date'], stock_df['close'], label='Close Price')
plt.plot(stock_df['date'], stock_df['SMA_7'], label='SMA 7', color='red')
plt.title('Stock Close Price & SMA 7')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.savefig('stock_sma.png')
plt.close()

# Plot RSI
plt.figure(figsize=(12,5))
plt.plot(stock_df['date'], stock_df['RSI_14'], label='RSI 14', color='purple')
plt.title('RSI 14 Over Time')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.tight_layout()
plt.savefig('rsi_plot.png')
plt.close()

# -------------------------------
# Step 5: Task 3 - Correlation Analysis
# -------------------------------

# Compute daily returns
daily_returns = calculate_daily_returns(stock_df)

# Align dates for sentiment and returns
aligned_returns = daily_returns[daily_sentiment.index.min():daily_sentiment.index.max()]

# Compute correlation
correlation = compute_correlation(daily_sentiment, aligned_returns)
print(f"\nCorrelation between daily sentiment & stock returns: {correlation:.3f}")

# Scatter plot for correlation
plt.figure(figsize=(8,5))
plt.scatter(daily_sentiment, aligned_returns, alpha=0.6, color='teal')
plt.title(f'Sentiment vs Daily Returns (Correlation: {correlation:.3f})')
plt.xlabel('Average Daily Sentiment')
plt.ylabel('Daily Stock Returns')
plt.tight_layout()
plt.savefig('sentiment_vs_returns.png')
plt.close()

print("\nAll plots saved. Final analysis complete.")
