"""
fetch_real_data.py
------------------
Fetches 100% REAL constituents, weights, market caps, and historical daily prices
for Euro Stoxx 50 and Nasdaq 100 using OpenBB / yfinance. No synthetic data.
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
    print("[1/4] Fetching live Nasdaq 100 constituents and official weights...")
    url = "https://www.slickcharts.com/nasdaq100"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    df = tables[0]
    
    # Standardize columns
    df = df.rename(columns={
        "Company": "company",
        "Symbol": "symbol",
        "Weight": "weight_str",
        "Price": "price"
    })
    
    # Parse weight string '12.54%' -> 0.1254
    df["index_weight"] = df["weight_str"].str.rstrip("%").astype(float) / 100.0
    # Clean symbol for yfinance (e.g. BRK.B to BRK-B if any, though NDX doesn't have BRK)
    df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)
    
    csv_path = os.path.join(DATA_DIR, "nasdaq100_constituents.csv")
    df.to_csv(csv_path, index=False)
    print(f" -> Saved {len(df)} Nasdaq 100 constituents to {csv_path}")
    return df

def fetch_eurostoxx50_constituents():
    print("[2/4] Fetching Euro Stoxx 50 constituents...")
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
        raise ValueError("Could not extract Euro Stoxx 50 table from Wikipedia")
        
    df_es50 = df_es50.rename(columns={
        sym_col: "symbol",
        "Name": "company",
        "Sector": "sector"
    })
    
    # Clean ticker symbols
    df_es50["symbol"] = df_es50["symbol"].str.strip()
    
    csv_path = os.path.join(DATA_DIR, "eurostoxx50_constituents.csv")
    df_es50.to_csv(csv_path, index=False)
    print(f" -> Saved {len(df_es50)} Euro Stoxx 50 constituents to {csv_path}")
    return df_es50

def download_prices_and_marketcaps(symbols, index_name, start_date="2021-01-01"):
    print(f"[3/4] Downloading real market data for {index_name} ({len(symbols)} tickers) from {start_date} to today...")
    
    # 1. Download adjusted close prices
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
        
    # Check completeness
    valid_cols = prices.columns[prices.notna().sum() > len(prices) * 0.70].tolist()
    prices = prices[valid_cols].ffill().dropna()
    
    price_path = os.path.join(DATA_DIR, f"{index_name}_prices.csv")
    prices.to_csv(price_path)
    print(f" -> Saved clean historical prices: shape {prices.shape} to {price_path}")
    
    # 2. Fetch current real market caps with ThreadPoolExecutor
    print(f" -> Fetching real market capitalizations for {index_name} in parallel...")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    market_caps = {}
    def get_mc(sym):
        try:
            t = yf.Ticker(sym)
            mc = getattr(t.fast_info, 'market_cap', None)
            if mc is None or np.isnan(mc):
                mc = t.info.get('marketCap', None)
            return sym, mc
        except Exception:
            return sym, None

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(get_mc, sym) for sym in valid_cols]
        for f in as_completed(futures):
            sym, mc = f.result()
            market_caps[sym] = mc
            
    mc_df = pd.DataFrame(list(market_caps.items()), columns=["symbol", "market_cap"])
    mc_path = os.path.join(DATA_DIR, f"{index_name}_market_caps.csv")
    mc_df.to_csv(mc_path, index=False)
    print(f" -> Saved {len(mc_df)} real market caps to {mc_path}")
    
    return prices, mc_df

def main():
    print("=================================================================")
    print(" REAL DATA ACQUISITION: NASDAQ 100 & EURO STOXX 50 (NO SYNTHETIC)")
    print("=================================================================")
    
    # 1. Nasdaq 100
    ndx_df = fetch_nasdaq100_constituents()
    ndx_symbols = ndx_df["symbol"].tolist()
    ndx_prices, ndx_mc = download_prices_and_marketcaps(ndx_symbols, "nasdaq100", start_date="2021-01-01")
    
    # 2. Euro Stoxx 50
    es50_df = fetch_eurostoxx50_constituents()
    es50_symbols = es50_df["symbol"].tolist()
    es50_prices, es50_mc = download_prices_and_marketcaps(es50_symbols, "eurostoxx50", start_date="2021-01-01")
    
    print("\n[COMPLETE] Real data successfully fetched and stored in:")
    print(f" -> {DATA_DIR}")

if __name__ == "__main__":
    main()
