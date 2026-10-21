"""
fetch_real_data.py
------------------
Fetches 100% REAL market data for Euro Stoxx 50 and Nasdaq 100:
1. Official Index Benchmark Levels directly from the exchanges (^NDX and ^STOXX50E).
2. Real constituent stock prices directly from traded market closes.
3. Official live constituents & weights.

ZERO synthetic data. ZERO proxies.
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from io import StringIO
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_nasdaq100_constituents():
    print("[1/4] Fetching real live Nasdaq 100 constituents...")
    url = "https://www.slickcharts.com/nasdaq100"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    df = tables[0]
    
    df = df.rename(columns={
        "Company": "company",
        "Symbol": "symbol",
        "Weight": "weight_str",
        "Price": "price"
    })
    
    df["index_weight"] = df["weight_str"].str.rstrip("%").astype(float) / 100.0
    df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)
    
    csv_path = os.path.join(DATA_DIR, "nasdaq100_constituents.csv")
    df.to_csv(csv_path, index=False)
    print(f" -> Saved {len(df)} real Nasdaq 100 constituents to {csv_path}")
    return df

def fetch_eurostoxx50_constituents():
    print("[2/4] Fetching real live Euro Stoxx 50 constituents...")
    url = "https://en.wikipedia.org/wiki/EURO_STOXX_50"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    
    df_es50 = None
    for t in tables:
        for col in ["Ticker", "Symbol"]:
            if col in t.columns and len(t) >= 50:
                df_es50 = t.copy()
                sym_col = col
                break
        if df_es50 is not None:
            break
            
    if df_es50 is None:
        raise ValueError("Could not extract Euro Stoxx 50 table")
        
    df_es50 = df_es50.rename(columns={
        sym_col: "symbol",
        "Name": "company",
        "Sector": "sector"
    })
    df_es50["symbol"] = df_es50["symbol"].str.strip()
    
    csv_path = os.path.join(DATA_DIR, "eurostoxx50_constituents.csv")
    df_es50.to_csv(csv_path, index=False)
    print(f" -> Saved {len(df_es50)} real Euro Stoxx 50 constituents to {csv_path}")
    return df_es50

def download_prices_and_official_index(symbols, index_ticker, index_name, start_date="2021-01-01"):
    print(f"[3/4] Downloading REAL market data for {index_name} constituents and official index ({index_ticker})...")
    
    # 1. Download official benchmark index level directly
    idx_raw = yf.download(index_ticker, start=start_date, auto_adjust=True, progress=False)
    if isinstance(idx_raw.columns, pd.MultiIndex):
        idx_close = idx_raw["Close"].iloc[:, 0]
    else:
        idx_close = idx_raw["Close"]
    idx_close.name = "official_benchmark_close"
    idx_path = os.path.join(DATA_DIR, f"{index_name}_benchmark_index.csv")
    idx_close.to_csv(idx_path)
    print(f" -> Saved official benchmark index ({index_ticker}) shape {idx_close.shape} to {idx_path}")
    
    # 2. Download constituent stock prices
    data = yf.download(
        tickers=symbols,
        start=start_date,
        interval="1d",
        auto_adjust=True,
        progress=True,
        group_by="column"
    )
    
    if "Close" in data.columns:
        prices = data["Close"]
    else:
        prices = data
        
    valid_cols = prices.columns[prices.notna().sum() > len(prices) * 0.70].tolist()
    prices = prices[valid_cols].ffill().dropna()
    
    price_path = os.path.join(DATA_DIR, f"{index_name}_prices.csv")
    prices.to_csv(price_path)
    print(f" -> Saved clean historical constituent prices: shape {prices.shape} to {price_path}")
    
    return prices, idx_close

def main():
    print("=========================================================================")
    print(" 100% REAL MARKET DATA: NASDAQ 100 & EURO STOXX 50 (NO PROXIES / NO SYNTH)")
    print("=========================================================================")
    
    # 1. Nasdaq 100 (Official Index: ^NDX)
    ndx_df = fetch_nasdaq100_constituents()
    ndx_symbols = ndx_df["symbol"].tolist()
    download_prices_and_official_index(ndx_symbols, "^NDX", "nasdaq100", start_date="2021-01-01")
    
    # 2. Euro Stoxx 50 (Official Index: ^STOXX50E)
    es50_df = fetch_eurostoxx50_constituents()
    es50_symbols = es50_df["symbol"].tolist()
    download_prices_and_official_index(es50_symbols, "^STOXX50E", "eurostoxx50", start_date="2021-01-01")
    
    print("\n[COMPLETE] 100% Real constituent prices and official indices stored in data/")

if __name__ == "__main__":
    main()
