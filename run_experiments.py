"""
run_experiments.py
------------------
Modular Out-of-Sample Backtest Script comparing Official Index Benchmarks
(^NDX for Nasdaq 100, ^STOXX50E for Euro Stoxx 50) against Re-Engineered Weightings:
- Equal Weight (1/N)
- Scenario-based Downside Risk / Minimum CVaR (Rockafellar & Uryasev 2000)
- Black-Litterman Shrinkage
- Worst-Case Robust Optimization (SOCP)

Evaluated with Second-Order Stochastic Dominance (SSD) Spanning Tests
(Arvanitis, Hallam, Post, & Topaloglou, 2019).
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from portfolio_models import (
    get_equal_weights,
    get_min_cvar_weights,
    get_black_litterman_weights,
    get_robust_weights
)
from stochastic_spanning import arvanitis_topaloglou_subsampling_test

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def compute_performance_metrics(returns_series, rf=0.02 / 252):
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
    bench_file = os.path.join(DATA_DIR, f"{index_name}_benchmark_index.csv")
    
    prices = pd.read_csv(price_file, index_col=0, parse_dates=True)
    bench_raw = pd.read_csv(bench_file, index_col=0, parse_dates=True)
    
    # Align dates between constituent stocks and official benchmark index
    common_dates = prices.index.intersection(bench_raw.index)
    prices = prices.loc[common_dates].ffill().dropna()
    bench_series = bench_raw.loc[common_dates].iloc[:, 0].ffill().dropna()
    
    valid_dates = prices.index.intersection(bench_series.index)
    prices = prices.loc[valid_dates]
    bench_series = bench_series.loc[valid_dates]
    
    returns = prices.pct_change().dropna()
    bench_returns = bench_series.pct_change().loc[returns.index]
    
    N = prices.shape[1]
    T = len(returns)
    print(f"Aligned Data: {T} daily periods across {N} real traded stocks.")
    
    strategies = [
        "Official Benchmark Index",
        "Equal Weight (1/N)",
        "Min CVaR (Downside)",
        "Black-Litterman",
        "Robust Optimization"
    ]
    strat_returns = {s: [] for s in strategies}
    current_weights = {
        "Equal Weight (1/N)": get_equal_weights(N),
        "Min CVaR (Downside)": get_equal_weights(N),
        "Black-Litterman": get_equal_weights(N),
        "Robust Optimization": get_equal_weights(N)
    }
    
    print(f"Executing rolling out-of-sample rebalancing (Lookback={lookback}d, Rebalance={rebalance_freq}d)...")
    
    for t in range(lookback, T):
        if (t - lookback) % rebalance_freq == 0:
            hist_returns = returns.iloc[t - lookback : t].values
            current_weights["Equal Weight (1/N)"] = get_equal_weights(N)
            current_weights["Min CVaR (Downside)"] = get_min_cvar_weights(hist_returns, alpha=0.95)
            current_weights["Black-Litterman"] = get_black_litterman_weights(hist_returns, get_equal_weights(N))
            current_weights["Robust Optimization"] = get_robust_weights(hist_returns, kappa=0.15)
            
        daily_r = returns.iloc[t].values
        strat_returns["Official Benchmark Index"].append(bench_returns.iloc[t])
        for s in ["Equal Weight (1/N)", "Min CVaR (Downside)", "Black-Litterman", "Robust Optimization"]:
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
    
    # 2. Stochastic Dominance Spanning Tests
    print("\n--- SECOND-ORDER STOCHASTIC DOMINANCE (SSD) SPANNING TESTS ---")
    print("Testing H0: Official Benchmark Index spans/dominates alternative weighting strategy")
    bench_r = df_out["Official Benchmark Index"].values
    
    sd_results = []
    for s in strategies:
        if s == "Official Benchmark Index":
            continue
        test_r = df_out[s].values
        test_res = arvanitis_topaloglou_subsampling_test(
            test_r, bench_r, n_subsamples=400, num_grid=60
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
    es50_metrics, es50_sd = run_backtest_for_index("eurostoxx50", rebalance_freq=21, lookback=252)
    ndx_metrics, ndx_sd = run_backtest_for_index("nasdaq100", rebalance_freq=21, lookback=252)
    print("\n[ALL EXPERIMENTS COMPLETED SUCCESSFULLY]")

if __name__ == "__main__":
    main()
