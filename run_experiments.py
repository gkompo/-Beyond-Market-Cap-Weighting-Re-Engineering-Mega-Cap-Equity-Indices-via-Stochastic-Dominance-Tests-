"""
run_experiments.py
------------------
Main execution script for Thesis Research:
Re-Engineering Equity Indices (Nasdaq 100 & Euro Stoxx 50) using Real Data.
Compares Market-Cap Benchmark against:
- Equal Weight (1/N)
- Scenario-based Downside Risk / Minimum CVaR
- Black-Litterman
- Worst-Case Robust Optimization

Then evaluates each strategy with Second-Order Stochastic Dominance (SSD)
Spanning Tests (Arvanitis, Hallam, Post, & Topaloglou, 2019).
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from portfolio_models import (
    get_market_cap_weights,
    get_equal_weights,
    get_min_cvar_weights,
    get_black_litterman_weights,
    get_robust_weights
)
from stochastic_spanning import arvanitis_topaloglou_subsampling_test

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def compute_performance_metrics(returns_series, rf=0.02 / 252):
    """Computes comprehensive risk and return metrics on daily returns."""
    r = returns_series.values
    ann_factor = 252
    
    ann_return = np.mean(r) * ann_factor
    ann_vol = np.std(r, ddof=1) * np.sqrt(ann_factor)
    sharpe = (ann_return - (rf * ann_factor)) / (ann_vol + 1e-8)
    
    downside_r = r[r < 0]
    downside_std = np.std(downside_r, ddof=1) * np.sqrt(ann_factor) if len(downside_r) > 0 else 1e-8
    sortino = (ann_return - (rf * ann_factor)) / (downside_std + 1e-8)
    
    cum_returns = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - peak) / peak
    max_drawdown = np.min(drawdowns)
    
    var_95 = np.percentile(r, 5)
    cvar_95 = np.mean(r[r <= var_95])
    
    return {
        "Annualized Return": f"{ann_return * 100:.2f}%",
        "Annualized Volatility": f"{ann_vol * 100:.2f}%",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Sortino Ratio": f"{sortino:.2f}",
        "Max Drawdown": f"{max_drawdown * 100:.2f}%",
        "Daily VaR (95%)": f"{var_95 * 100:.2f}%",
        "Daily CVaR (95%)": f"{cvar_95 * 100:.2f}%"
    }

def run_backtest_for_index(index_name, rebalance_freq=21, lookback=252):
    print(f"\n=======================================================")
    print(f" RUNNING RE-ENGINEERING BACKTEST: {index_name.upper()}")
    print(f"=======================================================")
    
    price_file = os.path.join(DATA_DIR, f"{index_name}_prices.csv")
    mc_file = os.path.join(DATA_DIR, f"{index_name}_market_caps.csv")
    
    prices = pd.read_csv(price_file, index_col=0, parse_dates=True)
    mc_df = pd.read_csv(mc_file)
    
    # Align constituents
    common_tickers = [c for c in prices.columns if c in mc_df["symbol"].values]
    prices = prices[common_tickers].ffill().dropna()
    mc_series = mc_df.set_index("symbol").loc[common_tickers, "market_cap"]
    
    # Daily returns
    returns = prices.pct_change().dropna()
    T, N = returns.shape
    print(f"Aligned Data: {T} daily periods across {N} real stocks.")
    
    # Market-Cap Benchmark weights
    mc_weights = get_market_cap_weights(mc_series)
    
    # Rolling out-of-sample simulation
    strategies = ["Market-Cap (Benchmark)", "Equal Weight (1/N)", "Min CVaR (Downside)", "Black-Litterman", "Robust Optimization"]
    strat_returns = {s: [] for s in strategies}
    rebalance_dates = []
    
    print(f"Executing rolling out-of-sample rebalancing (Lookback={lookback}d, Rebalance={rebalance_freq}d)...")
    
    # Start after warm-up
    current_weights = {
        "Market-Cap (Benchmark)": mc_weights,
        "Equal Weight (1/N)": get_equal_weights(N),
        "Min CVaR (Downside)": get_equal_weights(N),
        "Black-Litterman": mc_weights,
        "Robust Optimization": get_equal_weights(N)
    }
    
    for t in range(lookback, T):
        # Rebalance check
        if (t - lookback) % rebalance_freq == 0:
            hist_returns = returns.iloc[t - lookback : t].values
            
            # 1. Market-Cap
            current_weights["Market-Cap (Benchmark)"] = mc_weights
            # 2. Equal Weight
            current_weights["Equal Weight (1/N)"] = get_equal_weights(N)
            # 3. Min CVaR
            current_weights["Min CVaR (Downside)"] = get_min_cvar_weights(hist_returns, alpha=0.95)
            # 4. Black-Litterman
            current_weights["Black-Litterman"] = get_black_litterman_weights(hist_returns, mc_weights)
            # 5. Robust Optimization
            current_weights["Robust Optimization"] = get_robust_weights(hist_returns, kappa=0.15)
            
        daily_r = returns.iloc[t].values
        for s in strategies:
            strat_returns[s].append(np.dot(daily_r, current_weights[s]))
            
    df_out = pd.DataFrame(strat_returns, index=returns.index[lookback:])
    
    # 1. Performance Metrics Table
    metrics_list = []
    for s in strategies:
        m = compute_performance_metrics(df_out[s])
        m["Strategy"] = s
        metrics_list.append(m)
    metrics_df = pd.DataFrame(metrics_list).set_index("Strategy")
    print("\n--- OUT-OF-SAMPLE PERFORMANCE SUMMARY ---")
    print(metrics_df.to_string())
    
    # 2. Stochastic Dominance Spanning Tests (Topaloglou Methodology)
    print("\n--- SECOND-ORDER STOCHASTIC DOMINANCE (SSD) SPANNING TESTS ---")
    print("Testing H0: Market-Cap Benchmark spans/dominates alternative weighting strategy")
    bench_returns = df_out["Market-Cap (Benchmark)"].values
    
    sd_results = []
    for s in strategies:
        if s == "Market-Cap (Benchmark)":
            continue
        test_r = df_out[s].values
        test_res = arvanitis_topaloglou_subsampling_test(
            test_r, bench_returns, n_subsamples=400, num_grid=60
        )
        sd_results.append({
            "Alternative Strategy": s,
            "Eta Stat (Max Excess Downside Utility)": f"{test_res['obs_eta']:.5f}",
            "P-Value": f"{test_res['p_value']:.4f}",
            "SSD Dominance / Spanning": "H0 Rejected (Dominates Bmk)" if test_res['strictly_dominates'] else ("Excess Utility (eta > 0)" if test_res['obs_eta'] > 0 else "Benchmark Spans (H0 Accepted)")
        })
        
    sd_df = pd.DataFrame(sd_results).set_index("Alternative Strategy")
    print(sd_df.to_string())
    
    return metrics_df, sd_df

def main():
    # 1. Run for Euro Stoxx 50
    es50_metrics, es50_sd = run_backtest_for_index("eurostoxx50", rebalance_freq=21, lookback=252)
    
    # 2. Run for Nasdaq 100
    ndx_metrics, ndx_sd = run_backtest_for_index("nasdaq100", rebalance_freq=21, lookback=252)
    
    print("\n[ALL EXPERIMENTS COMPLETED SUCCESSFULLY]")

if __name__ == "__main__":
    main()
