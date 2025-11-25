import pandas as pd

def load_news_data(path: str) -> pd.DataFrame:
    """
    Load financial news CSV data.
    
    Args:
        path (str): Path to the news CSV file.
    
    Returns:
        pd.DataFrame: DataFrame with columns: headline, date, publisher, stock.
    """
    df = pd.read_csv(path, parse_dates=['date'])
    # Ensure proper column names
    df.columns = [col.lower() for col in df.columns]
    return df

def load_stock_data(path: str) -> pd.DataFrame:
    """
    Load stock price CSV data.
    
    Args:
        path (str): Path to the stock CSV file.
    
    Returns:
        pd.DataFrame: DataFrame with columns: date, open, high, low, close, volume.
    """
    df = pd.read_csv(path, parse_dates=['date'])
    df.columns = [col.lower() for col in df.columns]
    return df
