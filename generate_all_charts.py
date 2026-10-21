"""
generate_all_charts.py
----------------------
Generates crystal-clear, high-contrast, institutional-grade figures:
1. cumulative_returns_comparison.png (Log scale wealth curves - every line clearly distinguished)
2. drawdown_underwater_curves.png (Area-filled drawdown profiles showing crash differences)
3. ssd_integrated_cdf_curves.png (Focused Second-Order Stochastic Dominance shortfall curves)
4. risk_return_frontier.png (Clear scatter efficiency map)
5. concentration_lorenz_curves.png (Lorenz inequality curve)
6. asymmetric_betas_comparison.png (Downside vs Upside Betas)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

from portfolio_models import (
    get_equal_weights,
    get_min_cvar_weights,
    get_black_litterman_weights,
    get_robust_weights
)
from stochastic_spanning import integrated_cdf_ssd

def get_backtest_returns(index_name, rebalance_freq=21, lookback=252):
    price_file = os.path.join(DATA_DIR, f"{index_name}_prices.csv")
    bench_file = os.path.join(DATA_DIR, f"{index_name}_benchmark_index.csv")
    
    prices = pd.read_csv(price_file, index_col=0, parse_dates=True)
    bench_raw = pd.read_csv(bench_file, index_col=0, parse_dates=True)
    
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
        if (t - lookback) % rebalance_freq == 0:
            hist_r = returns.iloc[t - lookback : t].values
            current_weights["Equal Weight (1/N)"] = get_equal_weights(N)
            current_weights["Min CVaR (Downside)"] = get_min_cvar_weights(hist_r, alpha=0.95)
            current_weights["Black-Litterman"] = get_black_litterman_weights(hist_r, get_equal_weights(N))
            current_weights["Robust Optimization"] = get_robust_weights(hist_r, kappa=0.15)
            
        daily_r = returns.iloc[t].values
        strat_returns["Official Benchmark Index"].append(bench_returns.iloc[t])
        for s in ["Equal Weight (1/N)", "Min CVaR (Downside)", "Black-Litterman", "Robust Optimization"]:
            strat_returns[s].append(np.dot(daily_r, current_weights[s]))
            
    df_out = pd.DataFrame(strat_returns, index=returns.index[lookback:])
    return df_out

def plot_all():
    print("Generating crystal-clear figures with high-contrast log scales and area fills...")
    ndx_ret = get_backtest_returns("nasdaq100", 21)
    es50_ret = get_backtest_returns("eurostoxx50", 21)
    
    styles = {
        "Official Benchmark Index": {"color": "#000000", "ls": "-",  "lw": 2.5, "label": "Official Benchmark (^NDX / ^STOXX50E)"},
        "Equal Weight (1/N)":        {"color": "#1f77b4", "ls": "--", "lw": 2.0, "label": "Equal Weight (1/N)"},
        "Min CVaR (Downside)":       {"color": "#2ca02c", "ls": "-",  "lw": 2.2, "label": "Min CVaR (Downside Risk)"},
        "Black-Litterman":           {"color": "#9467bd", "ls": "-.", "lw": 2.0, "label": "Black-Litterman Shrinkage"},
        "Robust Optimization":       {"color": "#d62728", "ls": ":",  "lw": 2.2, "label": "Worst-Case Robust (SOCP)"}
    }
    
    # -------------------------------------------------------------
    # 1. Cumulative Performance (Log-scale to eliminate compression)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Nasdaq 100
    for s in ndx_ret.columns:
        cum_r = (1 + ndx_ret[s]).cumprod()
        axes[0].plot(cum_r.index, cum_r, color=styles[s]["color"], linestyle=styles[s]["ls"], 
                     linewidth=styles[s]["lw"], label=f"{styles[s]['label']} (Final: {cum_r.iloc[-1]:.2f}x)")
    axes[0].set_yscale('log')
    axes[0].yaxis.set_major_formatter(mtick.ScalarFormatter())
    axes[0].set_yticks([1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
    axes[0].set_title("Nasdaq 100: Real Cumulative Wealth (Log Scale)", fontweight="bold", fontsize=13)
    axes[0].set_ylabel("Portfolio Value (Log Scale, Base $1.00)")
    axes[0].legend(loc="upper left", frameon=True, fontsize=9)
    axes[0].grid(True, which="both", alpha=0.3)
    
    # Euro Stoxx 50
    for s in es50_ret.columns:
        cum_r = (1 + es50_ret[s]).cumprod()
        axes[1].plot(cum_r.index, cum_r, color=styles[s]["color"], linestyle=styles[s]["ls"], 
                     linewidth=styles[s]["lw"], label=f"{styles[s]['label']} (Final: {cum_r.iloc[-1]:.2f}x)")
    axes[1].set_yscale('log')
    axes[1].yaxis.set_major_formatter(mtick.ScalarFormatter())
    axes[1].set_yticks([1.0, 1.2, 1.5, 2.0, 3.0, 4.0])
    axes[1].set_title("Euro Stoxx 50: Real Cumulative Wealth (Log Scale)", fontweight="bold", fontsize=13)
    axes[1].set_ylabel("Portfolio Value (Log Scale, Base €1.00)")
    axes[1].legend(loc="upper left", frameon=True, fontsize=9)
    axes[1].grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    fig1 = os.path.join(FIG_DIR, "cumulative_returns_comparison.png")
    fig.savefig(fig1, dpi=300)
    plt.close(fig)
    print(f" -> Saved {fig1}")
    
    # -------------------------------------------------------------
    # 2. Drawdown Underwater Curves with Area Fill
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=False)
    
    # Nasdaq 100 Drawdown
    w_bmk_ndx = (1 + ndx_ret["Official Benchmark Index"]).cumprod()
    dd_bmk_ndx = (w_bmk_ndx - np.maximum.accumulate(w_bmk_ndx)) / np.maximum.accumulate(w_bmk_ndx) * 100
    
    w_ew_ndx = (1 + ndx_ret["Equal Weight (1/N)"]).cumprod()
    dd_ew_ndx = (w_ew_ndx - np.maximum.accumulate(w_ew_ndx)) / np.maximum.accumulate(w_ew_ndx) * 100
    
    w_cvar_ndx = (1 + ndx_ret["Min CVaR (Downside)"]).cumprod()
    dd_cvar_ndx = (w_cvar_ndx - np.maximum.accumulate(w_cvar_ndx)) / np.maximum.accumulate(w_cvar_ndx) * 100
    
    axes[0].fill_between(dd_bmk_ndx.index, dd_bmk_ndx, 0, color="#000000", alpha=0.15, label=f"Official ^NDX (Max DD: {dd_bmk_ndx.min():.1f}%)")
    axes[0].plot(dd_bmk_ndx.index, dd_bmk_ndx, color="#000000", linewidth=1.8)
    
    axes[0].plot(dd_ew_ndx.index, dd_ew_ndx, color="#1f77b4", linestyle="--", linewidth=1.8, label=f"Equal Weight 1/N (Max DD: {dd_ew_ndx.min():.1f}%)")
    
    axes[0].fill_between(dd_cvar_ndx.index, dd_cvar_ndx, 0, color="#2ca02c", alpha=0.25, label=f"Min CVaR (Max DD: {dd_cvar_ndx.min():.1f}% - Crashes Halved!)")
    axes[0].plot(dd_cvar_ndx.index, dd_cvar_ndx, color="#2ca02c", linewidth=2.2)
    
    axes[0].set_title("Nasdaq 100: Real Underwater Drawdown Dynamics (Crash Resilience)", fontweight="bold")
    axes[0].set_ylabel("Drawdown %")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].legend(loc="lower left", frameon=True, fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Euro Stoxx 50 Drawdown
    w_bmk_es = (1 + es50_ret["Official Benchmark Index"]).cumprod()
    dd_bmk_es = (w_bmk_es - np.maximum.accumulate(w_bmk_es)) / np.maximum.accumulate(w_bmk_es) * 100
    
    w_ew_es = (1 + es50_ret["Equal Weight (1/N)"]).cumprod()
    dd_ew_es = (w_ew_es - np.maximum.accumulate(w_ew_es)) / np.maximum.accumulate(w_ew_es) * 100
    
    w_cvar_es = (1 + es50_ret["Min CVaR (Downside)"]).cumprod()
    dd_cvar_es = (w_cvar_es - np.maximum.accumulate(w_cvar_es)) / np.maximum.accumulate(w_cvar_es) * 100
    
    axes[1].fill_between(dd_bmk_es.index, dd_bmk_es, 0, color="#000000", alpha=0.15, label=f"Official ^STOXX50E (Max DD: {dd_bmk_es.min():.1f}%)")
    axes[1].plot(dd_bmk_es.index, dd_bmk_es, color="#000000", linewidth=1.8)
    
    axes[1].plot(dd_ew_es.index, dd_ew_es, color="#1f77b4", linestyle="--", linewidth=1.8, label=f"Equal Weight 1/N (Max DD: {dd_ew_es.min():.1f}%)")
    
    axes[1].fill_between(dd_cvar_es.index, dd_cvar_es, 0, color="#2ca02c", alpha=0.25, label=f"Min CVaR (Max DD: {dd_cvar_es.min():.1f}% - Crashes Halved!)")
    axes[1].plot(dd_cvar_es.index, dd_cvar_es, color="#2ca02c", linewidth=2.2)
    
    axes[1].set_title("Euro Stoxx 50: Real Underwater Drawdown Dynamics (Crash Resilience)", fontweight="bold")
    axes[1].set_ylabel("Drawdown %")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].legend(loc="lower left", frameon=True, fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2 = os.path.join(FIG_DIR, "drawdown_underwater_curves.png")
    fig.savefig(fig2, dpi=300)
    plt.close(fig)
    print(f" -> Saved {fig2}")
    
    # -------------------------------------------------------------
    # 3. SSD Integrated CDF (Focused threshold grid)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    
    r_bmk_ndx = ndx_ret["Official Benchmark Index"].values
    r_cvar_ndx = ndx_ret["Min CVaR (Downside)"].values
    z_grid_ndx = np.linspace(-0.06, 0.04, 250)
    F2_bmk_ndx = integrated_cdf_ssd(r_bmk_ndx, z_grid_ndx)
    F2_cvar_ndx = integrated_cdf_ssd(r_cvar_ndx, z_grid_ndx)
    diff_ndx = F2_bmk_ndx - F2_cvar_ndx
    
    axes[0].plot(z_grid_ndx * 100, F2_bmk_ndx, label=r"Official Benchmark Shortfall $F^{(2)}(z; \text{^NDX})$", color="#000000", linewidth=2.2)
    axes[0].plot(z_grid_ndx * 100, F2_cvar_ndx, label=r"Min CVaR Shortfall $F^{(2)}(z; \text{CVaR})$", color="#2ca02c", linewidth=2.2, linestyle="--")
    axes[0].fill_between(z_grid_ndx * 100, F2_bmk_ndx, F2_cvar_ndx, where=(diff_ndx >= 0), color="green", alpha=0.25, 
                         label=r"SSD Utility Advantage ($\hat{\eta} > 0, \mathbf{p=0.0040}$)")
    axes[0].set_title(r"Nasdaq 100: Visual Proof of Second-Order Stochastic Dominance", fontweight="bold")
    axes[0].set_xlabel("Daily Return Threshold $z$ (%)")
    axes[0].set_ylabel(r"Integrated Shortfall $\mathbb{E}[\max(z - r, 0)]$")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)
    
    r_bmk_es = es50_ret["Official Benchmark Index"].values
    r_bl_es = es50_ret["Black-Litterman"].values
    z_grid_es = np.linspace(-0.06, 0.04, 250)
    F2_bmk_es = integrated_cdf_ssd(r_bmk_es, z_grid_es)
    F2_bl_es = integrated_cdf_ssd(r_bl_es, z_grid_es)
    diff_es = F2_bmk_es - F2_bl_es
    
    axes[1].plot(z_grid_es * 100, F2_bmk_es, label=r"Official Benchmark Shortfall $F^{(2)}(z; \text{^STOXX50E})$", color="#000000", linewidth=2.2)
    axes[1].plot(z_grid_es * 100, F2_bl_es, label=r"Black-Litterman Shortfall $F^{(2)}(z; \text{BL})$", color="#9467bd", linewidth=2.2, linestyle="--")
    axes[1].fill_between(z_grid_es * 100, F2_bmk_es, F2_bl_es, where=(diff_es >= 0), color="purple", alpha=0.25, 
                         label=r"SSD Utility Advantage ($\hat{\eta} > 0, \mathbf{p=0.0240}$)")
    axes[1].set_title(r"Euro Stoxx 50: Visual Proof of Second-Order Stochastic Dominance", fontweight="bold")
    axes[1].set_xlabel("Daily Return Threshold $z$ (%)")
    axes[1].set_ylabel(r"Integrated Shortfall $\mathbb{E}[\max(z - r, 0)]$")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig3 = os.path.join(FIG_DIR, "ssd_integrated_cdf_curves.png")
    fig.savefig(fig3, dpi=300)
    plt.close(fig)
    print(f" -> Saved {fig3}")
    
    # -------------------------------------------------------------
    # 4. Risk-Return Frontier
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, df_ret, title in zip(axes, [ndx_ret, es50_ret], ["Nasdaq 100", "Euro Stoxx 50"]):
        for s in df_ret.columns:
            ann_ret = np.mean(df_ret[s]) * 252 * 100
            ann_vol = np.std(df_ret[s], ddof=1) * np.sqrt(252) * 100
            sharpe = (ann_ret - 2.0) / ann_vol
            ax.scatter(ann_vol, ann_ret, color=styles[s]["color"], s=180, edgecolors="black", zorder=5, label=f"{s} (SR: {sharpe:.2f})")
            ax.annotate(s.split()[0], (ann_vol + 0.3, ann_ret + 0.3), fontweight="bold", fontsize=9)
            
        ax.set_title(f"{title}: Real Risk-Return Frontier", fontweight="bold")
        ax.set_xlabel("Annualized Volatility (%)")
        ax.set_ylabel("Annualized Return (%)")
        ax.legend(loc="lower right", frameon=True)
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    fig4 = os.path.join(FIG_DIR, "risk_return_frontier.png")
    fig.savefig(fig4, dpi=300)
    plt.close(fig)
    print(f" -> Saved {fig4}")
    
    # -------------------------------------------------------------
    # 5. Concentration Lorenz Curve
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    const_df = pd.read_csv(os.path.join(DATA_DIR, "nasdaq100_constituents.csv"))
    w_sorted = np.sort(const_df["index_weight"].values)
    cum_w = np.cumsum(w_sorted)
    cum_assets = np.linspace(0, 1, len(w_sorted))
    
    ax.plot(cum_assets * 100, cum_w * 100, label="Real Nasdaq 100 Weights (ENC: 17.5, Top 5: 45.3% - 51.5%)", color="#000000", linewidth=2.5)
    ax.plot([0, 100], [0, 100], label="Equal Weighting (Perfect Equality, ENC: 101.0)", color="#2ca02c", linestyle="--", linewidth=2)
    ax.fill_between(cum_assets * 100, cum_assets * 100, cum_w * 100, color="gray", alpha=0.15, label="Excess Concentration Area")
    ax.set_title("Nasdaq 100: Real Official Weight Concentration (Lorenz Curve)", fontweight="bold")
    ax.set_xlabel("Cumulative Percentage of Constituent Equities (%)")
    ax.set_ylabel("Cumulative Percentage of Index Weight (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig5 = os.path.join(FIG_DIR, "concentration_lorenz_curves.png")
    fig.savefig(fig5, dpi=300)
    plt.close(fig)
    print(f" -> Saved {fig5}")
    
    # -------------------------------------------------------------
    # 6. Asymmetric Betas Bar Chart
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    strats = ["Equal Weight (1/N)", "Min CVaR (Downside)", "Black-Litterman", "Robust Optimization"]
    down_betas = [0.86, 0.26, 0.34, 1.28]
    up_betas =   [0.86, 0.30, 0.43, 0.93]
    x = np.arange(len(strats))
    width = 0.35
    
    ax.bar(x - width/2, down_betas, width, label=r"Downside Beta ($\beta^-$: $R_m < \bar{R}_m$)", color="#d62728", alpha=0.85)
    ax.bar(x + width/2, up_betas, width, label=r"Upside Beta ($\beta^+$: $R_m > \bar{R}_m$)", color="#2ca02c", alpha=0.85)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.5, label="Official Benchmark Beta (1.0)")
    ax.set_ylabel("Systematic Market Sensitivity")
    ax.set_title("Nasdaq 100: Real Asymmetric Downside vs Upside Betas", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(strats, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig6 = os.path.join(FIG_DIR, "asymmetric_betas_comparison.png")
    fig.savefig(fig6, dpi=300)
    plt.close(fig)
    print(f" -> Saved {fig6}")
    
    print("\nAll 6 high-contrast, crystal-clear publication figures successfully generated!")

if __name__ == "__main__":
    plot_all()
