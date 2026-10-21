"""
run_full_econometric_study.py
-----------------------------
Comprehensive Master Empirical Study for Diploma Thesis:
- Data: 100% Real constituent data from 2021 to 2026 (~5.8 years, ~4.8 years out-of-sample)
- Indices: Euro Stoxx 50 and Nasdaq 100
- Frequencies: Monthly (21d), Bimonthly (42d), Quarterly (63d)
- Frictionless Assumption: Zero transaction costs and spreads
- Models: Market-Cap, Equal Weight (1/N), Min CVaR, Black-Litterman, Robust Optimization
- Econometric Batteries:
    1. Risk-adjusted metrics (Sharpe, Sortino, Max Drawdown, VaR 95%, CVaR 95%)
    2. Distributional Diagnostics (Jarque-Bera normality, ARCH-LM clustering)
    3. CAPM Regressions with Newey-West (HAC) standard errors (Alpha, Beta, t-stat)
    4. Asymmetric Downside Beta (Ang, Chen, Xing 2006) vs Upside Beta
    5. Ledoit-Wolf / Memmel Sharpe Ratio Difference Tests
    6. Second-Order Stochastic Dominance (SSD) Spanning Tests (Topaloglou Methodology)
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
    get_market_cap_weights,
    get_equal_weights,
    get_min_cvar_weights,
    get_black_litterman_weights,
    get_robust_weights
)
from stochastic_spanning import arvanitis_topaloglou_subsampling_test

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def downside_upside_betas(r_test, r_bench):
    """Downside (r_b < mean) and Upside (r_b > mean) betas (Ang et al., 2006)."""
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
    """Jobson-Korkie with Memmel (2003) correction."""
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
    
    var95 = np.percentile(r, 5)
    cvar95 = np.mean(r[r <= var95])
    
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
    mc_file = os.path.join(DATA_DIR, f"{index_name}_market_caps.csv")
    
    prices = pd.read_csv(price_file, index_col=0, parse_dates=True)
    mc_df = pd.read_csv(mc_file)
    common_tickers = [c for c in prices.columns if c in mc_df["symbol"].values]
    prices = prices[common_tickers].ffill().dropna()
    returns = prices.pct_change().dropna()
    
    mc_series = mc_df.set_index("symbol").loc[common_tickers, "market_cap"]
    mc_weights = get_market_cap_weights(mc_series)
    N = len(common_tickers)
    T = len(returns)
    
    strategies = [
        "Market-Cap (Benchmark)",
        "Equal Weight (1/N)",
        "Min CVaR (Downside)",
        "Black-Litterman",
        "Robust Optimization"
    ]
    
    strat_returns = {s: [] for s in strategies}
    current_weights = {
        "Market-Cap (Benchmark)": mc_weights,
        "Equal Weight (1/N)": get_equal_weights(N),
        "Min CVaR (Downside)": get_equal_weights(N),
        "Black-Litterman": mc_weights,
        "Robust Optimization": get_equal_weights(N)
    }
    
    for t in range(lookback, T):
        if (t - lookback) % rebalance_freq == 0:
            hist_r = returns.iloc[t - lookback : t].values
            current_weights["Market-Cap (Benchmark)"] = mc_weights
            current_weights["Equal Weight (1/N)"] = get_equal_weights(N)
            current_weights["Min CVaR (Downside)"] = get_min_cvar_weights(hist_r, alpha=0.95)
            current_weights["Black-Litterman"] = get_black_litterman_weights(hist_r, mc_weights)
            current_weights["Robust Optimization"] = get_robust_weights(hist_r, kappa=0.15)
            
        daily_r = returns.iloc[t].values
        for s in strategies:
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
    print(f" FULL EMPIRICAL STUDY FOR: {index_name.upper()} (REAL DATA)")
    print("="*80)
    
    # Read raw prices to show exact time span
    prices = pd.read_csv(os.path.join(DATA_DIR, f"{index_name}_prices.csv"), index_col=0, parse_dates=True)
    start_dt = prices.index[0].strftime('%Y-%m-%d')
    end_dt = prices.index[-1].strftime('%Y-%m-%d')
    total_days = len(prices)
    oos_days = total_days - 252
    years_total = total_days / 252.0
    years_oos = oos_days / 252.0
    
    print(f"Data Sample Period: {start_dt} to {end_dt} ({total_days} bars = {years_total:.1f} years total)")
    print(f"Warm-Up Lookback:   252 trading days (1.0 year)")
    print(f"Out-of-Sample Test: {prices.index[252].strftime('%Y-%m-%d')} to {end_dt} ({oos_days} bars = {years_oos:.1f} years)")
    print("Microstructure:     Frictionless Assumption (Zero transaction costs & zero spreads)\n")
    
    all_summary_tables = {}
    
    for freq_label, freq_days in freqs.items():
        print(f"\n>>> EVALUATING REBALANCING FREQUENCY: {freq_label.upper()} <<<")
        df_returns = run_simulation(index_name, rebalance_freq=freq_days)
        r_bmk = df_returns["Market-Cap (Benchmark)"].values
        
        # 1. Performance Summary
        perf_rows = []
        for s in df_returns.columns:
            m = compute_metrics(df_returns[s])
            m["Strategy"] = s
            perf_rows.append(m)
        perf_df = pd.DataFrame(perf_rows).set_index("Strategy")
        print("\n[A] Performance & Risk Profile:")
        print(perf_df.to_string())
        
        # 2. Econometric Regressions (CAPM HAC, Downside Beta, Sharpe diff, SSD test)
        reg_rows = []
        for s in df_returns.columns:
            if s == "Market-Cap (Benchmark)":
                continue
            r_s = df_returns[s].values
            
            # CAPM with Newey-West HAC
            X = sm.add_constant(r_bmk)
            model = sm.OLS(r_s, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
            alpha_ann = model.params[0] * 252
            alpha_t = model.tvalues[0]
            alpha_p = model.pvalues[0]
            mkt_beta = model.params[1]
            r2 = model.rsquared
            
            # Asymmetric Betas
            d_beta, u_beta = downside_upside_betas(r_s, r_bmk)
            
            # Sharpe difference test
            _, z_sr, p_sr = ledoit_wolf_sharpe_test(r_s, r_bmk)
            
            # Topaloglou SSD Spanning test (fast subsampling: 250 iterations)
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
        print("\n[B] Econometric Factor Regressions, Asymmetric Betas & Topaloglou Spanning Tests:")
        print(reg_df.to_string())
        all_summary_tables[freq_label] = (perf_df, reg_df)

if __name__ == "__main__":
    analyze_index_all_frequencies("eurostoxx50")
    analyze_index_all_frequencies("nasdaq100")
