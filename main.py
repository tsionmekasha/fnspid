# main.py
"""
Final submission for FNSPID Project
Tasks:
1. EDA + Keyword Extraction
2. Sentiment Analysis
3. Technical Analysis (TA-Lib + PyNance)
4. Correlation Analysis
"""

from src.pipelines import run_eda_pipeline, run_correlation_pipeline
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Run EDA Pipeline
# -------------------------------
news_csv_path = "data/news.csv"
news_df, daily_sentiment, top_keywords = run_eda_pipeline(news_csv_path)

# Save top keywords plot
plt.figure(figsize=(12,6))
top_keywords.plot(kind='bar')
plt.title("Top 20 Keywords in Headlines")
plt.xlabel("Keywords")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("plots/top_keywords.png")
plt.close()

# -------------------------------
# Step 2: Run Correlation & Quantitative Analysis Pipeline
# -------------------------------
stock_csv_path = "data/stock.csv"
stock_df, daily_returns, volatility, correlation = run_correlation_pipeline(stock_csv_path, daily_sentiment)

# Plot daily sentiment vs stock returns
plt.figure(figsize=(10,6))
plt.scatter(daily_sentiment, daily_returns)
plt.title("Daily Sentiment vs Stock Returns")
plt.xlabel("Average Daily Sentiment")
plt.ylabel("Daily Stock Return")
plt.tight_layout()
plt.savefig("plots/sentiment_vs_returns.png")
plt.close()

# Plot volatility
plt.figure(figsize=(10,6))
volatility.plot()
plt.title("Rolling Volatility of Stock")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.tight_layout()
plt.savefig("plots/volatility.png")
plt.close()

print("Final submission pipelines executed successfully!")
print(f"Correlation between sentiment & stock returns: {correlation:.3f}")
