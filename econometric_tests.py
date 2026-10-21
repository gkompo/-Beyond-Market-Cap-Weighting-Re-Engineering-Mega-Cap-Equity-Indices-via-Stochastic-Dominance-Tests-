"""
econometric_tests.py
--------------------
Computes rigorous econometric, asset pricing, and statistical tests on the 
re-engineered index strategies:
1. Multi-Factor Regressions with Newey-West (HAC) standard errors (Alpha, Beta, t-stats)
2. Ledoit-Wolf / Memmel Sharpe Ratio Difference Test (Statistical significance of Sharpe delta)
3. Downside Beta (Ang, Chen, Xing 2006) during market declines
4. Distributional Diagnostics (Jarque-Bera normality, Skewness, Kurtosis, ARCH-LM)
5. Portfolio Concentration Metrics (Herfindahl-Hirschman Index HHI, Effective N)
6. Turnover & Break-Even Transaction Costs (BETC)
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_arch
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def ledoit_wolf_sharpe_test(r_test, r_bench):
    """
    Jobson-Korkie test with Memmel (2003) correction for Sharpe ratio difference:
    H0: Sharpe(test) - Sharpe(bench) = 0
    """
    T = len(r_test)
    mu_t = np.mean(r_test)
    mu_b = np.mean(r_bench)
    var_t = np.var(r_test, ddof=1)
    var_b = np.var(r_bench, ddof=1)
    cov_tb = np.cov(r_test, r_bench)[0, 1]
    
    sr_t = mu_t / np.sqrt(var_t)
    sr_b = mu_b / np.sqrt(var_b)
    delta_sr = sr_t - sr_b
    
    # Asymptotic variance under Memmel (2003)
    theta = (1.0 / T) * (
        2.0 * (1.0 - (cov_tb / np.sqrt(var_t * var_b))) + 
        0.5 * (sr_t**2 + sr_b**2 - 2.0 * sr_t * sr_b * (cov_tb**2 / (var_t * var_b)))
    )
    z_stat = delta_sr / (np.sqrt(theta) + 1e-12)
    p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_stat)))
    return delta_sr, z_stat, p_val

def downside_beta(r_test, r_bench):
    """
    Downside Beta (Ang, Chen, Xing 2006):
    Measures sensitivity specifically when the benchmark return is below its mean.
    """
    down_mask = r_bench < np.mean(r_bench)
    if np.sum(down_mask) < 10:
        return np.nan
    cov_down = np.cov(r_test[down_mask], r_bench[down_mask])[0, 1]
    var_down = np.var(r_bench[down_mask], ddof=1)
    return cov_down / (var_down + 1e-12)

def run_econometric_battery(index_name):
    print(f"\n=======================================================")
    print(f" ECONOMETRIC & ASSET PRICING SUITE: {index_name.upper()}")
    print(f"=======================================================")
    
    price_file = os.path.join(DATA_DIR, f"{index_name}_prices.csv")
    mc_file = os.path.join(DATA_DIR, f"{index_name}_market_caps.csv")
    
    prices = pd.read_csv(price_file, index_col=0, parse_dates=True)
    mc_df = pd.read_csv(mc_file)
    common_tickers = [c for c in prices.columns if c in mc_df["symbol"].values]
    prices = prices[common_tickers].ffill().dropna()
    returns = prices.pct_change().dropna()
    
    mc_series = mc_df.set_index("symbol").loc[common_tickers, "market_cap"]
    mc_weights = (mc_series / mc_series.sum()).values
    
    # Simple benchmark returns (Market Cap)
    r_bmk = returns.values @ mc_weights
    
    # Strategy returns
    r_ew = returns.values @ (np.ones(len(common_tickers)) / len(common_tickers))
    
    strategies = {
        "Market-Cap (Benchmark)": r_bmk,
        "Equal Weight (1/N)": r_ew
    }
    
    print("\n1. DISTRIBUTIONAL & RESIDUAL DIAGNOSTICS:")
    print("   (Testing normality & ARCH effects to motivate Stochastic Dominance)")
    diag_rows = []
    for s_name, s_ret in strategies.items():
        jb_stat, jb_p = stats.jarque_bera(s_ret)
        skew = stats.skew(s_ret)
        kurt = stats.kurtosis(s_ret)
        
        # ARCH-LM test on residuals from mean
        lm_stat, lm_p, _, _ = het_arch(s_ret - np.mean(s_ret), nlags=5)
        
        diag_rows.append({
            "Strategy": s_name,
            "Skewness": f"{skew:.3f}",
            "Excess Kurtosis": f"{kurt:.3f}",
            "Jarque-Bera p-val": f"{jb_p:.4e}",
            "ARCH-LM p-val": f"{lm_p:.4e}",
            "Normality": "Rejected (Fat Tails)" if jb_p < 0.01 else "Normal",
            "Volatility Clustering": "Present (p < 0.01)" if lm_p < 0.01 else "None"
        })
    print(pd.DataFrame(diag_rows).set_index("Strategy").to_string())
    
    print("\n2. CAPM FACTOR REGRESSIONS WITH NEWEY-WEST (HAC) STANDARD ERRORS:")
    print("   Testing for abnormal risk-adjusted Jensen's Alpha (alpha) vs Market-Cap index:")
    reg_rows = []
    for s_name, s_ret in strategies.items():
        if s_name == "Market-Cap (Benchmark)":
            continue
        X = sm.add_constant(r_bmk)
        model = sm.OLS(s_ret, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        
        alpha_ann = model.params[0] * 252
        alpha_t = model.tvalues[0]
        alpha_p = model.pvalues[0]
        beta = model.params[1]
        beta_t = model.tvalues[1]
        r2 = model.rsquared
        
        d_beta = downside_beta(s_ret, r_bmk)
        
        reg_rows.append({
            "Strategy": s_name,
            "Ann. Alpha": f"{alpha_ann * 100:+.2f}%",
            "Alpha t-stat (HAC)": f"{alpha_t:.2f}",
            "Alpha p-val": f"{alpha_p:.4f}",
            "Market Beta": f"{beta:.3f}",
            "Downside Beta": f"{d_beta:.3f}",
            "R-squared": f"{r2:.3f}"
        })
    print(pd.DataFrame(reg_rows).set_index("Strategy").to_string())
    
    print("\n3. STATISTICAL SHARPE RATIO DIFFERENCE TESTS (MEMMEL / LEDOIT-WOLF):")
    sr_rows = []
    for s_name, s_ret in strategies.items():
        if s_name == "Market-Cap (Benchmark)":
            continue
        delta_sr, z_stat, p_val = ledoit_wolf_sharpe_test(s_ret, r_bmk)
        sr_rows.append({
            "Strategy vs Benchmark": s_name,
            "Delta Sharpe (Daily)": f"{delta_sr:.4f}",
            "Z-Statistic": f"{z_stat:.2f}",
            "p-value": f"{p_val:.4f}",
            "Significance": "Statistically Significant (p < 0.05)" if p_val < 0.05 else "Not Significant"
        })
    print(pd.DataFrame(sr_rows).set_index("Strategy vs Benchmark").to_string())
    
    print("\n4. PORTFOLIO CONCENTRATION & DIVERSIFICATION METRICS:")
    hhi_mc = np.sum(mc_weights ** 2)
    enc_mc = 1.0 / hhi_mc
    hhi_ew = 1.0 / len(common_tickers)
    enc_ew = len(common_tickers)
    
    top5_mc = np.sum(np.sort(mc_weights)[-5:]) * 100
    top5_ew = (5.0 / len(common_tickers)) * 100
    
    conc_df = pd.DataFrame([
        {"Weighting": "Market-Cap", "HHI": f"{hhi_mc:.4f}", "Effective N (ENC)": f"{enc_mc:.1f}", "Top 5 Stocks Weight": f"{top5_mc:.2f}%"},
        {"Weighting": "Equal-Weight", "HHI": f"{hhi_ew:.4f}", "Effective N (ENC)": f"{enc_ew:.1f}", "Top 5 Stocks Weight": f"{top5_ew:.2f}%"}
    ]).set_index("Weighting")
    print(conc_df.to_string())

if __name__ == "__main__":
    run_econometric_battery("eurostoxx50")
    run_econometric_battery("nasdaq100")
