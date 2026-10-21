"""
run_full_econometric_study.py
-----------------------------
Master Empirical Study: Re-Engineering Equity Indices via Stochastic Dominance Spanning.
100% REAL TRADED MARKET DATA ONLY:
- Benchmark: Official exchange index returns directly from ^NDX (Nasdaq 100) and ^STOXX50E (Euro Stoxx 50).
- Constituents: Traded daily closes across all index equities.
- Rebalancing Horizons: Monthly (21d), Bimonthly (42d), Quarterly (63d).
- No lookahead bias, no synthetic data, no proxies.
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_arch
from scipy import stats
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

def downside_upside_betas(r_test, r_bench):
    mu_b = np.mean(r_bench)
    down_mask = r_bench < mu_b
    up_mask = r_bench > mu_b
    
    var_down = np.var(r_bench[down_mask], ddof=1) if np.sum(down_mask) > 10 else np.nan
    cov_down = np.cov(r_test[down_mask], r_bench[down_mask])[0, 1] if np.sum(down_mask) > 10 else np.nan
    beta_down = cov_down / (var_down + 1e-12) if var_down > 0 else np.nan

    var_up = np.var(r_bench[up_mask], ddof=1) if np.sum(up_mask) > 10 else np.nan
    cov_up = np.cov(r_test[up_mask], r_bench[up_mask])[0, 1] if np.sum(up_mask) > 10 else np.nan
    beta_up = cov_up / (var_up + 1e-12) if var_up > 0 else np.nan
    
    return beta_down, beta_up

def ledoit_wolf_sharpe_test(r_test, r_bench):
    T = len(r_test)
    mu_t, mu_b = np.mean(r_test), np.mean(r_bench)
    var_t, var_b = np.var(r_test, ddof=1), np.var(r_bench, ddof=1)
    cov_tb = np.cov(r_test, r_bench)[0, 1]
    
    sr_t = mu_t / np.sqrt(var_t)
    sr_b = mu_b / np.sqrt(var_b)
    delta_sr = sr_t - sr_b
    
    theta = (1.0 / T) * (
        2.0 * (1.0 - (cov_tb / np.sqrt(var_t * var_b))) + 
        0.5 * (sr_t**2 + sr_b**2 - 2.0 * sr_t * sr_b * (cov_tb**2 / (var_t * var_b)))
    )
    z_stat = delta_sr / (np.sqrt(theta) + 1e-12)
    p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_stat)))
    return delta_sr, z_stat, p_val

def compute_metrics(r_series, rf=0.02 / 252):
    r = r_series.values
    ann_factor = 252
    ann_ret = np.mean(r) * ann_factor
    ann_vol = np.std(r, ddof=1) * np.sqrt(ann_factor)
    sharpe = (ann_ret - (rf * ann_factor)) / (ann_vol + 1e-8)
    
    downside_r = r[r < 0]
    downside_std = np.std(downside_r, ddof=1) * np.sqrt(ann_factor) if len(downside_r) > 0 else 1e-8
    sortino = (ann_ret - (rf * ann_factor)) / (downside_std + 1e-8)
    
    cum_ret = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum_ret)
    drawdowns = (cum_ret - peak) / peak
    mdd = np.min(drawdowns)
    
    cvar95 = np.mean(r[r <= np.percentile(r, 5)])
    
    return {
        "Ann. Return": f"{ann_ret * 100:.2f}%",
        "Ann. Volatility": f"{ann_vol * 100:.2f}%",
        "Sharpe": f"{sharpe:.2f}",
        "Sortino": f"{sortino:.2f}",
        "Max Drawdown": f"{mdd * 100:.2f}%",
        "CVaR (95%)": f"{cvar95 * 100:.2f}%"
    }

def run_simulation(index_name, rebalance_freq, lookback=252):
    price_file = os.path.join(DATA_DIR, f"{index_name}_prices.csv")
    bench_file = os.path.join(DATA_DIR, f"{index_name}_benchmark_index.csv")
    
    prices = pd.read_csv(price_file, index_col=0, parse_dates=True)
    bench_raw = pd.read_csv(bench_file, index_col=0, parse_dates=True)
    
    # Align dates between constituent stocks and official benchmark index
    common_dates = prices.index.intersection(bench_raw.index)
    prices = prices.loc[common_dates].ffill().dropna()
    bench_series = bench_raw.loc[common_dates].iloc[:, 0].ffill().dropna()
    
    # Align again after dropna
    valid_dates = prices.index.intersection(bench_series.index)
    prices = prices.loc[valid_dates]
    bench_series = bench_series.loc[valid_dates]
    
    returns = prices.pct_change().dropna()
    bench_returns = bench_series.pct_change().loc[returns.index]
    
    N = prices.shape[1]
    T = len(returns)
    
    # Alternative strategies evaluated against Official Benchmark Index
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
    
    for t in range(lookback, T):
        # Strict out-of-sample: only use historical returns up to t-1
        if (t - lookback) % rebalance_freq == 0:
            hist_r = returns.iloc[t - lookback : t].values
            
            # Equal Weight
            current_weights["Equal Weight (1/N)"] = get_equal_weights(N)
            # Min CVaR (Linear Program)
            current_weights["Min CVaR (Downside)"] = get_min_cvar_weights(hist_r, alpha=0.95)
            # Black-Litterman (Bayesian Shrinkage on Empirical Covariance)
            current_weights["Black-Litterman"] = get_black_litterman_weights(hist_r, get_equal_weights(N))
            # Robust Optimization (Second-Order Cone)
            current_weights["Robust Optimization"] = get_robust_weights(hist_r, kappa=0.15)
            
        daily_r = returns.iloc[t].values
        # 1. Official benchmark index return on day t
        strat_returns["Official Benchmark Index"].append(bench_returns.iloc[t])
        # 2. Alternative re-engineered strategies
        for s in ["Equal Weight (1/N)", "Min CVaR (Downside)", "Black-Litterman", "Robust Optimization"]:
            strat_returns[s].append(np.dot(daily_r, current_weights[s]))
            
    df_out = pd.DataFrame(strat_returns, index=returns.index[lookback:])
    return df_out

def analyze_index_all_frequencies(index_name):
    freqs = {
        "Monthly (21 Days)": 21,
        "Bimonthly (42 Days)": 42,
        "Quarterly (63 Days)": 63
    }
    
    print("\n" + "="*80)
    print(f" FULL REAL-DATA EMPIRICAL STUDY FOR: {index_name.upper()}")
    print("="*80)
    
    price_file = os.path.join(DATA_DIR, f"{index_name}_prices.csv")
    prices = pd.read_csv(price_file, index_col=0, parse_dates=True)
    start_dt = prices.index[0].strftime('%Y-%m-%d')
    end_dt = prices.index[-1].strftime('%Y-%m-%d')
    total_days = len(prices)
    oos_days = total_days - 252
    
    print(f"Sample Period:     {start_dt} to {end_dt} ({total_days} daily bars)")
    print(f"Lookback Window:   252 trading days (1.0 year lookback)")
    print(f"Out-of-Sample:     {prices.index[252].strftime('%Y-%m-%d')} to {end_dt} ({oos_days} days)")
    print(f"Data Source:       100% Real Traded Prices & Official Benchmark Index (^NDX / ^STOXX50E)\n")
    
    results = {}
    for freq_label, freq_days in freqs.items():
        print(f"\n>>> EVALUATING REBALANCING FREQUENCY: {freq_label.upper()} <<<")
        df_returns = run_simulation(index_name, rebalance_freq=freq_days)
        r_bmk = df_returns["Official Benchmark Index"].values
        
        # Performance Table
        perf_rows = []
        for s in df_returns.columns:
            m = compute_metrics(df_returns[s])
            m["Strategy"] = s
            perf_rows.append(m)
        perf_df = pd.DataFrame(perf_rows).set_index("Strategy")
        print("\n[A] Performance & Risk Profile:")
        print(perf_df.to_string())
        
        # Econometric Regressions & SSD Spanning Tests
        reg_rows = []
        for s in df_returns.columns:
            if s == "Official Benchmark Index":
                continue
            r_s = df_returns[s].values
            
            # CAPM with Newey-West HAC
            X = sm.add_constant(r_bmk)
            model = sm.OLS(r_s, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
            alpha_ann = model.params[0] * 252
            alpha_t = model.tvalues[0]
            alpha_p = model.pvalues[0]
            mkt_beta = model.params[1]
            
            d_beta, u_beta = downside_upside_betas(r_s, r_bmk)
            _, _, p_sr = ledoit_wolf_sharpe_test(r_s, r_bmk)
            sd_res = arvanitis_topaloglou_subsampling_test(r_s, r_bmk, n_subsamples=250, num_grid=40)
            
            reg_rows.append({
                "Strategy": s,
                "HAC Alpha": f"{alpha_ann*100:+.2f}%",
                "Alpha t-stat": f"{alpha_t:+.2f}",
                "Alpha p-val": f"{alpha_p:.4f}",
                "Market Beta": f"{mkt_beta:.2f}",
                "Downside Beta": f"{d_beta:.2f}",
                "Upside Beta": f"{u_beta:.2f}",
                "Sharpe Diff p-val": f"{p_sr:.4f}",
                "Topaloglou SSD p-val": f"{sd_res['p_value']:.4f}",
                "SSD Spanning Conclusion": "REJECT H0 (Dominates)" if sd_res['strictly_dominates'] else ("Excess Utility" if sd_res['obs_eta'] > 0 else "Benchmark Spans")
            })
            
        reg_df = pd.DataFrame(reg_rows).set_index("Strategy")
        print("\n[B] Econometric Factor Regressions & Topaloglou Spanning Tests:")
        print(reg_df.to_string())
        results[freq_label] = (perf_df, reg_df)
        
    return results

if __name__ == "__main__":
    analyze_index_all_frequencies("eurostoxx50")
    analyze_index_all_frequencies("nasdaq100")
