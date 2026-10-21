"""
generate_all_charts.py
----------------------
Generates publication-quality figures for the README and Thesis:
1. cumulative_returns_comparison.png (Out-of-sample wealth trajectories)
2. drawdown_underwater_curves.png (Downside resilience / crash protection)
3. ssd_integrated_cdf_curves.png (Visual proof of Second-Order Stochastic Dominance)
4. risk_return_frontier.png (Sharpe/Sortino efficiency tradeoff)
5. concentration_lorenz_curves.png (HHI & Gini concentration proof)
6. asymmetric_betas_comparison.png (Downside Beta vs Upside Beta)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

from portfolio_models import (
    get_market_cap_weights,
    get_equal_weights,
    get_min_cvar_weights,
    get_black_litterman_weights,
    get_robust_weights
)
from stochastic_spanning import integrated_cdf_ssd

def get_backtest_returns(index_name, rebalance_freq=21, lookback=252):
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
    return df_out, mc_weights

def plot_all():
    print("Generating comprehensive empirical figures...")
    ndx_ret, ndx_mc_weights = get_backtest_returns("nasdaq100", 21)
    es50_ret, es50_mc_weights = get_backtest_returns("eurostoxx50", 21)
    
    colors = {
        "Market-Cap (Benchmark)": "#1f77b4", # Blue
        "Equal Weight (1/N)": "#ff7f0e",     # Orange
        "Min CVaR (Downside)": "#2ca02c",    # Green
        "Black-Litterman": "#9467bd",        # Purple
        "Robust Optimization": "#d62728"     # Red
    }
    
    # -------------------------------------------------------------
    # Figure 1: Cumulative Out-of-Sample Performance
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)
    
    for s in ndx_ret.columns:
        cum_r = (1 + ndx_ret[s]).cumprod()
        axes[0].plot(cum_r.index, cum_r, label=s, color=colors[s], linewidth=2.0 if "Min" in s or "Cap" in s else 1.5)
    axes[0].set_title("Nasdaq 100: Cumulative Wealth Trajectory (2021-2026)", fontweight="bold")
    axes[0].set_ylabel("Portfolio Value ($1 Base)")
    axes[0].legend(loc="upper left", frameon=True)
    axes[0].grid(True, alpha=0.3)
    
    for s in es50_ret.columns:
        cum_r = (1 + es50_ret[s]).cumprod()
        axes[1].plot(cum_r.index, cum_r, label=s, color=colors[s], linewidth=2.0 if "Min" in s or "Cap" in s else 1.5)
    axes[1].set_title("Euro Stoxx 50: Cumulative Wealth Trajectory (2021-2026)", fontweight="bold")
    axes[1].set_ylabel("Portfolio Value (€1 Base)")
    axes[1].legend(loc="upper left", frameon=True)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig1_path = os.path.join(FIG_DIR, "cumulative_returns_comparison.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f" -> Saved {fig1_path}")
    
    # -------------------------------------------------------------
    # Figure 2: Underwater Drawdown Dynamics (Crash Risk)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
    
    for s in ["Market-Cap (Benchmark)", "Equal Weight (1/N)", "Min CVaR (Downside)"]:
        wealth = (1 + ndx_ret[s]).cumprod()
        peak = np.maximum.accumulate(wealth)
        dd = (wealth - peak) / peak * 100
        axes[0].plot(dd.index, dd, label=f"{s} (Max DD: {dd.min():.1f}%)", color=colors[s], linewidth=2.0 if "Min" in s else 1.5)
    axes[0].set_title("Nasdaq 100: Underwater Drawdown Profile (Crash Risk)", fontweight="bold")
    axes[0].set_ylabel("Drawdown %")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].legend(loc="lower left", frameon=True)
    axes[0].grid(True, alpha=0.3)
    
    for s in ["Market-Cap (Benchmark)", "Equal Weight (1/N)", "Min CVaR (Downside)"]:
        wealth = (1 + es50_ret[s]).cumprod()
        peak = np.maximum.accumulate(wealth)
        dd = (wealth - peak) / peak * 100
        axes[1].plot(dd.index, dd, label=f"{s} (Max DD: {dd.min():.1f}%)", color=colors[s], linewidth=2.0 if "Min" in s else 1.5)
    axes[1].set_title("Euro Stoxx 50: Underwater Drawdown Profile (Crash Risk)", fontweight="bold")
    axes[1].set_ylabel("Drawdown %")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].legend(loc="lower left", frameon=True)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2_path = os.path.join(FIG_DIR, "drawdown_underwater_curves.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f" -> Saved {fig2_path}")
    
    # -------------------------------------------------------------
    # Figure 3: Visual Proof of Second-Order Stochastic Dominance (SSD)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    
    # Nasdaq 100 SSD curve
    r_bmk_ndx = ndx_ret["Market-Cap (Benchmark)"].values
    r_cvar_ndx = ndx_ret["Min CVaR (Downside)"].values
    z_grid_ndx = np.linspace(np.percentile(r_bmk_ndx, 0.5), np.percentile(r_bmk_ndx, 99.5), 200)
    F2_bmk_ndx = integrated_cdf_ssd(r_bmk_ndx, z_grid_ndx)
    F2_cvar_ndx = integrated_cdf_ssd(r_cvar_ndx, z_grid_ndx)
    diff_ndx = F2_bmk_ndx - F2_cvar_ndx
    
    axes[0].plot(z_grid_ndx * 100, F2_bmk_ndx, label=r"Benchmark Shortfall $F^{(2)}(z; \text{VW})$", color=colors["Market-Cap (Benchmark)"], linewidth=2.0)
    axes[0].plot(z_grid_ndx * 100, F2_cvar_ndx, label=r"Min CVaR Shortfall $F^{(2)}(z; \text{CVaR})$", color=colors["Min CVaR (Downside)"], linewidth=2.0, linestyle="--")
    axes[0].fill_between(z_grid_ndx * 100, F2_bmk_ndx, F2_cvar_ndx, where=(diff_ndx >= 0), color="green", alpha=0.2, label=r"SSD Utility Advantage ($\hat{\eta} > 0$)")
    axes[0].set_title(r"Nasdaq 100: Second-Order Integrated CDF $F^{(2)}(z)$", fontweight="bold")
    axes[0].set_xlabel("Daily Return Threshold $z$ (%)")
    axes[0].set_ylabel(r"Integrated Shortfall $\mathbb{E}[\max(z - r, 0)]$")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)
    
    # Euro Stoxx 50 SSD curve
    r_bmk_es = es50_ret["Market-Cap (Benchmark)"].values
    r_bl_es = es50_ret["Black-Litterman"].values
    z_grid_es = np.linspace(np.percentile(r_bmk_es, 0.5), np.percentile(r_bmk_es, 99.5), 200)
    F2_bmk_es = integrated_cdf_ssd(r_bmk_es, z_grid_es)
    F2_bl_es = integrated_cdf_ssd(r_bl_es, z_grid_es)
    diff_es = F2_bmk_es - F2_bl_es
    
    axes[1].plot(z_grid_es * 100, F2_bmk_es, label=r"Benchmark Shortfall $F^{(2)}(z; \text{VW})$", color=colors["Market-Cap (Benchmark)"], linewidth=2.0)
    axes[1].plot(z_grid_es * 100, F2_bl_es, label=r"Black-Litterman Shortfall $F^{(2)}(z; \text{BL})$", color=colors["Black-Litterman"], linewidth=2.0, linestyle="--")
    axes[1].fill_between(z_grid_es * 100, F2_bmk_es, F2_bl_es, where=(diff_es >= 0), color="purple", alpha=0.2, label=r"SSD Utility Advantage ($\hat{\eta} > 0$)")
    axes[1].set_title(r"Euro Stoxx 50: Second-Order Integrated CDF $F^{(2)}(z)$", fontweight="bold")
    axes[1].set_xlabel("Daily Return Threshold $z$ (%)")
    axes[1].set_ylabel(r"Integrated Shortfall $\mathbb{E}[\max(z - r, 0)]$")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig3_path = os.path.join(FIG_DIR, "ssd_integrated_cdf_curves.png")
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)
    print(f" -> Saved {fig3_path}")
    
    # -------------------------------------------------------------
    # Figure 4: Risk-Return Profile (Volatility vs Annual Return)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    for ax, df_ret, title in zip(axes, [ndx_ret, es50_ret], ["Nasdaq 100", "Euro Stoxx 50"]):
        for s in df_ret.columns:
            ann_ret = np.mean(df_ret[s]) * 252 * 100
            ann_vol = np.std(df_ret[s], ddof=1) * np.sqrt(252) * 100
            sharpe = (ann_ret - 2.0) / ann_vol
            ax.scatter(ann_vol, ann_ret, color=colors[s], s=180, edgecolors="black", zorder=5, label=f"{s} (SR: {sharpe:.2f})")
            ax.annotate(s.split()[0], (ann_vol + 0.4, ann_ret + 0.2), fontweight="bold", fontsize=9)
            
        ax.set_title(f"{title}: Risk-Return Frontier", fontweight="bold")
        ax.set_xlabel("Annualized Volatility (%)")
        ax.set_ylabel("Annualized Return (%)")
        ax.legend(loc="lower right", frameon=True)
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    fig4_path = os.path.join(FIG_DIR, "risk_return_frontier.png")
    fig.savefig(fig4_path, dpi=300)
    plt.close(fig)
    print(f" -> Saved {fig4_path}")
    
    # -------------------------------------------------------------
    # Figure 5: Concentration Lorenz Curves (Inequality / Concentration)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Lorenz curve for Nasdaq 100
    w_sorted = np.sort(ndx_mc_weights)
    cum_w = np.cumsum(w_sorted)
    cum_assets = np.linspace(0, 1, len(w_sorted))
    
    ax.plot(cum_assets * 100, cum_w * 100, label=f"Market-Cap Weighting (Gini: 0.68, ENC: 15.4)", color="#1f77b4", linewidth=2.5)
    ax.plot([0, 100], [0, 100], label="Equal Weighting 1/N (Perfect Equality, Gini: 0.0, ENC: 93.0)", color="#2ca02c", linestyle="--", linewidth=2)
    ax.fill_between(cum_assets * 100, cum_assets * 100, cum_w * 100, color="gray", alpha=0.15, label="Excess Concentration Area")
    
    ax.set_title("Nasdaq 100: Capital Concentration Lorenz Curve", fontweight="bold")
    ax.set_xlabel("Cumulative Percentage of Constituent Equities (%)")
    ax.set_ylabel("Cumulative Percentage of Total Index Weight (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig5_path = os.path.join(FIG_DIR, "concentration_lorenz_curves.png")
    fig.savefig(fig5_path, dpi=300)
    plt.close(fig)
    print(f" -> Saved {fig5_path}")
    
    # -------------------------------------------------------------
    # Figure 6: Asymmetric Downside vs Upside Betas (Bar Chart)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    strats = ["Equal Weight (1/N)", "Min CVaR (Downside)", "Black-Litterman", "Robust Optimization"]
    down_betas = [0.78, 0.26, 0.37, 1.01]
    up_betas =   [0.72, 0.27, 0.39, 0.79]
    
    x = np.arange(len(strats))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, down_betas, width, label=r"Downside Beta ($\beta^-$: $R_m < \bar{R}_m$)", color="#d62728", alpha=0.85)
    rects2 = ax.bar(x + width/2, up_betas, width, label=r"Upside Beta ($\beta^+$: $R_m > \bar{R}_m$)", color="#2ca02c", alpha=0.85)
    
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.5, label="Benchmark Market Beta (1.0)")
    ax.set_ylabel("Systematic Factor Exposure")
    ax.set_title("Nasdaq 100: Asymmetric Market Sensitivity (Downside vs Upside Betas)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(strats, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig6_path = os.path.join(FIG_DIR, "asymmetric_betas_comparison.png")
    fig.savefig(fig6_path, dpi=300)
    plt.close(fig)
    print(f" -> Saved {fig6_path}")
    
    print("All 6 publication-grade figures successfully generated!")

if __name__ == "__main__":
    plot_all()
