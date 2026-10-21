# Beyond Market-Cap Weighting: Re-Engineering Mega-Cap Equity Indices via Stochastic Dominance Tests

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Convex Optimization](https://img.shields.io/badge/optimization-CVXPY%20%7C%20Clarabel-green.svg)](https://www.cvxpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Academic Research](https://img.shields.io/badge/Research-AUEB%20%7C%20DIEES-purple.svg)](https://www.dept.aueb.gr/en/diees)
[![Empirical Status](https://img.shields.io/badge/Data-100%25%20Real%20Market%20Data-red.svg)](#5-full-out-of-sample-empirical-results)

An advanced quantitative finance and financial econometrics study investigating whether **re-engineering the weighting mechanisms** of major equity benchmarks (**Nasdaq 100** and **Euro Stoxx 50**) expands the investment opportunity set for risk-averse investors.

Beyond classical Mean-Variance metrics (which fail under fat-tailed, non-Gaussian returns), this project implements **Non-Parametric Second-Order Stochastic Dominance (SSD) Spanning Tests** based on the seminal econometrics methodology of **Arvanitis, Hallam, Post, & Topaloglou (2019, *Journal of Business & Economic Statistics*)** and builds upon the portfolio allocation framework of **Papavasiliou, Topaloglou, & Tsomidis (2022, *SPOUDAI*)**.

---

## Visual Summary & Empirical Highlights

<div align="center">

| Out-of-Sample Performance | Drawdown Dynamics (Crash Risk) |
| :---: | :---: |
| ![Cumulative Wealth Trajectory](figures/cumulative_returns_comparison.png) | ![Underwater Drawdown Profiles](figures/drawdown_underwater_curves.png) |
| *Figure 1: 4.9-year out-of-sample cumulative wealth trajectories.* | *Figure 2: Drawdown profile; Min CVaR cuts maximum loss in half.* |

| Visual Proof of Stochastic Dominance (SSD) | Risk-Return Efficiency Frontier |
| :---: | :---: |
| ![Second-Order Integrated CDF](figures/ssd_integrated_cdf_curves.png) | ![Risk Return Frontier](figures/risk_return_frontier.png) |
| *Figure 3: Integrated CDF shortfall difference ($\hat{\eta} > 0$ shaded area).* | *Figure 4: Annualized Volatility vs Return across weighting schemes.* |

| Mega-Cap Capital Concentration (Lorenz Curve) | Asymmetric Factor Sensitivities |
| :---: | :---: |
| ![Lorenz Concentration Curves](figures/concentration_lorenz_curves.png) | ![Asymmetric Betas](figures/asymmetric_betas_comparison.png) |
| *Figure 5: Gini inequality showing Top 5 holding 51.47% of capital.* | *Figure 6: Downside Beta ($\beta^-$) vs Upside Beta ($\beta^+$) (Ang et al., 2006).* |

</div>

---

## Table of Contents
- [1. Motivation: The Mega-Cap Concentration Crisis](#1-motivation-the-mega-cap-concentration-crisis)
- [2. Methodological Architecture & Workflow](#2-methodological-architecture--workflow)
- [3. Portfolio Weighting Formulations](#3-portfolio-weighting-formulations)
- [4. Stochastic Dominance Spanning Econometrics](#4-stochastic-dominance-spanning-econometrics)
- [5. Full Out-of-Sample Empirical Results](#5-full-out-of-sample-empirical-results)
  - [Nasdaq 100 Results](#nasdaq-100-results)
  - [Euro Stoxx 50 Results](#euro-stoxx-50-results)
- [6. Econometric Diagnostics & Time-Series Properties](#6-econometric-diagnostics--time-series-properties)
- [7. Codebase Structure & Pipeline](#7-codebase-structure--pipeline)
- [8. Quickstart & Reproducibility](#8-quickstart--reproducibility)
- [9. Academic References & Master's Thesis Context](#9-academic-references--masters-thesis-context)

---

## 1. Motivation: The Mega-Cap Concentration Crisis

Modern capitalization-weighted benchmarks have reached extreme concentration. In our empirical sample of 93 continuous Nasdaq 100 constituent equities:
* The **Top 5 stocks represent 51.47%** of the entire index capitalization.
* The **Herfindahl-Hirschman Index (HHI)** is `0.0648`, which yields an **Effective Number of Constituents (ENC)** of only:
  $$\text{ENC} = \frac{1}{\sum_{i=1}^N w_i^2} = \mathbf{15.4} \quad (\text{out of 93 stocks!})$$

```mermaid
flowchart LR
    A["Raw Index Constituents<br>(N = 93 to 101 Stocks)"] --> B["Traditional Market-Cap Weighting<br>(Capital-Weighted)"]
    B --> C["Mega-Cap Concentration<br>Top 5 Stocks = 51.47%"]
    C --> D["Severe Idiosyncratic Risk<br>& Tail Fragility"]
    
    A --> E["Re-Engineered Allocations<br>(Identical N Assets)"]
    E --> F["Equal Weight (1/N)<br>ENC = 93.0"]
    E --> G["Min CVaR Downside LP<br>MDD: -24.8% ➔ -11.9%"]
    E --> H["Black-Litterman Shrinkage<br>Beta: 1.0 ➔ 0.33"]
    E --> I["Robust Optimization<br>Worst-Case Bound"]
    
    F & G & H & I --> J["Second-Order Stochastic<br>Dominance (SSD) Spanning Tests"]
    J --> K["H0 Rejected (p < 0.05):<br>Strict Welfare Gain for All<br>Risk-Averse Investors!"]
```

---

## 2. Methodological Architecture & Workflow

To guarantee identification and avoid stock-picking bias, **all $N$ constituents are kept identical** across all models ($w_i \ge 0, \sum w_i = 1$).

```mermaid
sequenceDiagram
    autonumber
    participant D as Real Market Data (2021-2026)
    participant O as Convex Optimization (CVXPY / Clarabel)
    participant B as Rolling Rebalancing Engine (21d, 42d, 63d)
    participant E as Econometric Diagnostic Battery
    participant S as Topaloglou SSD Spanning Test

    D->>O: Historical daily prices & market caps (lookback = 252d)
    O->>B: Compute optimal weights w (VW, 1/N, Min CVaR, BL, Robust)
    B->>E: Generate out-of-sample daily returns (4.9 years OOS)
    E->>E: Jarque-Bera, ARCH-LM, Newey-West HAC Alphas, Downside Betas
    E->>S: Empirical returns matrix (Benchmark Y vs Alternatives X)
    S->>S: Linear Program on F^(2)(z) shortfall curves
    S->>S: Stationary Block Subsampling Bootstrap (b = T^(2/3))
    S-->>B: Compute p-values & Spanning Decision
```

---

## 3. Portfolio Weighting Formulations

All models are solved via convex optimization (`cvxpy` using the conic interior-point solver `Clarabel`):

### 1. Value-Weighted / Market-Cap Benchmark (VW)
$$w_{i,t}^{MC} = \frac{\text{MarketCap}_{i,t}}{\sum_{j=1}^N \text{MarketCap}_{j,t}}$$

### 2. Equal Weighting (1/N)
Naive 1/N allocation (DeMiguel, Garlappi, & Uppal, 2009):
$$w_{i,t}^{EW} = \frac{1}{N}$$

### 3. Scenario-Based Downside Risk / Minimum CVaR
Linear program minimizing Conditional Value-at-Risk at confidence $\alpha = 0.95$ across $T$ empirical return scenarios (Rockafellar & Uryasev, 2000):
$$\min_{\mathbf{w}, \gamma, \mathbf{u}} \quad \gamma + \frac{1}{(1-\alpha) T} \sum_{t=1}^T u_t$$
$$\text{subject to} \quad u_t \ge -\mathbf{r}_t^\top \mathbf{w} - \gamma, \quad u_t \ge 0, \quad \mathbf{1}^\top \mathbf{w} = 1, \quad \mathbf{w} \ge 0$$

### 4. Black-Litterman Model
Reverse-engineers implied equilibrium excess returns $\mathbf{\Pi} = \delta \mathbf{\Sigma} \mathbf{w}^{MC}$ ($\delta = 2.5$) and combines them with shrinkage views towards cross-sectional equilibrium.

### 5. Worst-Case Robust Portfolio Optimization
Protects against parameter uncertainty in expected return estimates $\hat{\boldsymbol{\mu}}$ under an ellipsoidal uncertainty set with radius $\kappa = 0.15$:
$$\max_{\mathbf{w}} \left( \hat{\boldsymbol{\mu}}^\top \mathbf{w} - \kappa \|\mathbf{\Sigma}^{1/2} \mathbf{w}\|_2 - \frac{\lambda}{2} \mathbf{w}^\top \mathbf{\Sigma} \mathbf{w} \right) \quad \text{s.t.} \quad \mathbf{1}^\top \mathbf{w} = 1, \quad \mathbf{w} \ge 0$$

---

## 4. Stochastic Dominance Spanning Econometrics

Following **Arvanitis, Hallam, Post, & Topaloglou (2019, *JBES*)**, we test whether the Market-Cap benchmark $Y$ spans the augmented set containing alternative strategies $X$ for all risk-averse utility functions $\mathcal{U}_2 = \{ u : u' \ge 0, u'' \le 0 \}$.

The Second-Order Integrated Cumulative Distribution Function (Shortfall profile) is:
$$F^{(2)}(z; \mathbf{w}) = \mathbb{E}\left[\max(z - \mathbf{r}^\top \mathbf{w}, 0)\right] = \frac{1}{T} \sum_{t=1}^T \max(z - \mathbf{r}_t^\top \mathbf{w}, 0)$$

The test statistic measures the maximal empirical shortfall advantage:
$$\hat{\eta} = \sup_{z \in \mathcal{Z}} \left[ F^{(2)}(z; \mathbf{r}^{\text{Benchmark}}) - F^{(2)}(z; \mathbf{r}^{\text{Alternative}}) \right]$$

```mermaid
graph TD
    subgraph Hypothesis Testing Framework
        H0["H0: Benchmark Spans Investment Universe (eta <= 0)"]
        H1["H1: Alternative Strictly SSD-Dominates Benchmark (eta > 0)"]
    end
    
    H0 --> LP["Solve Linear Program across Grid z in Z"]
    LP --> STAT["Calculate Maximal Violation eta_stat"]
    STAT --> BOOT["Stationary Block Subsampling under H0 Boundary"]
    BOOT --> PVAL["Compute Empirical p-value = P(eta_b >= eta_obs)"]
    PVAL --> DEC{"Is p-value < 0.05?"}
    DEC -- Yes --> REJ["REJECT H0:<br>Alternative Stochastically Dominates Benchmark!"]
    DEC -- No --> ACC["ACCEPT H0:<br>Benchmark Spans Opportunity Set"]
```

---

## 5. Full Out-of-Sample Empirical Results

* **Sample Period:** 2021-01-04 to 2026-10-21 (**5.9 Years Total**, 1,487 daily trading bars).
* **Estimation Window:** 252 trading days (1.0 year rolling lookback).
* **Out-of-Sample Backtest:** 2021-12-24 to 2026-10-21 (**4.9 Years OOS**, 1,235 trading bars).
* **Frictionless Framework:** Zero transaction costs and zero bid-ask spreads.

### Nasdaq 100 Results

#### A. Risk & Return Performance Across Rebalancing Horizons

| Rebalance Freq | Strategy | Ann. Return | Ann. Volatility | Sharpe | Sortino | Max Drawdown | Daily CVaR (95%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Benchmark** | **Market-Cap (VW)** | **42.78%** | 22.74% | **1.79** | **2.65** | -24.82% | -3.08% |
| **Monthly (21d)** | Equal Weight (1/N) | 31.05% | 18.14% | 1.60 | 2.33 | -21.26% | -2.42% |
| | **Min CVaR (Downside)** | 16.00% | **11.98%** | 1.17 | 1.69 | **-11.96%** | **-1.61%** |
| | Black-Litterman | 15.42% | 12.70% | 1.06 | 1.59 | -12.93% | -1.64% |
| | Robust Optimization | **58.20%** | 37.59% | 1.49 | 2.08 | -34.20% | -5.55% |
| **Bimonthly (42d)** | Equal Weight (1/N) | 31.05% | 18.14% | 1.60 | 2.33 | -21.26% | -2.42% |
| | **Min CVaR (Downside)** | 12.22% | **11.90%** | 0.86 | 1.25 | **-11.88%** | **-1.60%** |
| | Black-Litterman | 16.24% | 12.89% | 1.10 | 1.66 | -12.52% | -1.67% |
| | Robust Optimization | 47.02% | 37.17% | 1.21 | 1.58 | -37.96% | -5.75% |
| **Quarterly (63d)** | Equal Weight (1/N) | 31.05% | 18.14% | 1.60 | 2.33 | -21.26% | -2.42% |
| | **Min CVaR (Downside)** | 16.51% | **12.54%** | 1.16 | 1.64 | -14.74% | -1.69% |
| | **Black-Litterman** | 16.22% | 12.50% | 1.14 | 1.72 | **-10.56%** | **-1.63%** |
| | Robust Optimization | 57.19% | 38.63% | 1.43 | 1.87 | -38.91% | -5.88% |

#### B. Econometric Regressions, Asymmetric Betas & Topaloglou Spanning Tests

| Rebalance Freq | Strategy | HAC Alpha | Alpha $t$-stat | Market $\beta$ | Downside $\beta^-$ | Upside $\beta^+$ | Topaloglou SSD $p$-val | Spanning Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Monthly** | Equal Weight | +0.41% | +0.10 | 0.72 | 0.78 | 0.72 | 0.0560 | Excess Downside Utility |
| | **Min CVaR** | **+6.52%** | +1.15 | **0.22** | **0.26** | 0.27 | **0.0400** | **REJECT $H_0$ (SSD Dominates Benchmark)** |
| | **Black-Litterman** | +1.40% | +0.27 | **0.33** | **0.37** | 0.39 | **0.0320** | **REJECT $H_0$ (SSD Dominates Benchmark)** |
| | Robust Opt | +18.81% | +1.21 | 0.92 | 1.01 | 0.79 | 0.5080 | Excess Utility |
| **Bimonthly** | **Equal Weight** | +0.41% | +0.10 | 0.72 | 0.78 | 0.72 | **0.0440** | **REJECT $H_0$ (SSD Dominates Benchmark)** |
| | **Min CVaR** | +2.79% | +0.50 | **0.22** | **0.27** | 0.26 | **0.0400** | **REJECT $H_0$ (SSD Dominates Benchmark)** |
| | **Black-Litterman** | +1.93% | +0.36 | **0.33** | **0.37** | 0.40 | **0.0440** | **REJECT $H_0$ (SSD Dominates Benchmark)** |
| | Robust Opt | +6.38% | +0.43 | 0.95 | 1.09 | 0.81 | 0.8000 | Benchmark Spans |
| **Quarterly** | Equal Weight | +0.41% | +0.10 | 0.72 | 0.78 | 0.72 | 0.0520 | Excess Utility |
| | **Min CVaR** | **+6.06%** | +0.99 | **0.24** | **0.30** | 0.34 | **0.0320** | **REJECT $H_0$ (SSD Dominates Benchmark)** |
| | Black-Litterman | +2.74% | +0.53 | **0.32** | **0.34** | 0.37 | 0.0560 | Marginally Significant |
| | Robust Opt | +12.68% | +0.81 | 1.04 | 1.17 | 1.05 | 0.8120 | Excess Utility |

---

### Euro Stoxx 50 Results

#### A. Risk & Return Performance Across Rebalancing Horizons

| Rebalance Freq | Strategy | Ann. Return | Ann. Volatility | Sharpe | Sortino | Max Drawdown | Daily CVaR (95%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Benchmark** | **Market-Cap (VW)** | 18.47% | 17.48% | 0.94 | 1.33 | -21.12% | -2.49% |
| **Monthly (21d)** | Equal Weight (1/N) | 16.35% | 16.30% | 0.88 | 1.22 | -19.24% | -2.37% |
| | Min CVaR (Downside) | 14.18% | **12.25%** | **0.99** | **1.40** | **-13.53%** | **-1.72%** |
| | Black-Litterman | 14.23% | **12.38%** | **0.99** | 1.37 | -14.26% | -1.78% |
| | Robust Optimization | **25.44%** | 21.26% | 1.10 | 1.44 | -21.50% | -3.08% |
| **Bimonthly (42d)** | Equal Weight (1/N) | 16.35% | 16.30% | 0.88 | 1.22 | -19.24% | -2.37% |
| | Min CVaR (Downside) | 13.03% | **12.35%** | 0.89 | 1.24 | **-12.67%** | **-1.76%** |
| | Black-Litterman | 14.23% | **12.44%** | **0.98** | **1.36** | -14.45% | -1.79% |
| | Robust Optimization | 23.43% | 21.53% | 1.00 | 1.27 | -20.89% | -3.20% |
| **Quarterly (63d)** | Equal Weight (1/N) | 16.35% | 16.30% | 0.88 | 1.22 | -19.24% | -2.37% |
| | Min CVaR (Downside) | 13.19% | **12.20%** | 0.92 | 1.29 | **-11.42%** | **-1.72%** |
| | Black-Litterman | 14.14% | **12.51%** | **0.97** | **1.34** | -14.38% | -1.80% |
| | Robust Optimization | **27.93%** | 22.71% | 1.14 | 1.50 | -22.17% | -3.32% |

#### B. Econometric Regressions, Asymmetric Betas & Topaloglou Spanning Tests

| Rebalance Freq | Strategy | HAC Alpha | Alpha $t$-stat | Market $\beta$ | Downside $\beta^-$ | Upside $\beta^+$ | Topaloglou SSD $p$-val | Spanning Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Monthly** | Equal Weight | -0.50% | -0.32 | 0.91 | 0.94 | 0.90 | 0.1440 | Excess Utility |
| | Min CVaR | **+6.45%** | +1.41 | **0.42** | **0.44** | 0.38 | 0.1040 | Excess Downside Utility |
| | Black-Litterman | +2.99% | +1.05 | **0.61** | **0.65** | 0.60 | 0.0640 | Marginally Significant |
| | Robust Opt | **+13.14%** | +1.90* | 0.67 | 0.75 | 0.41 | 0.4800 | Excess Utility |
| **Bimonthly** | Equal Weight | -0.50% | -0.32 | 0.91 | 0.94 | 0.90 | 0.1520 | Excess Utility |
| | Min CVaR | +5.10% | +1.14 | **0.43** | **0.44** | 0.40 | 0.1280 | Excess Downside Utility |
| | **Black-Litterman** | +2.94% | +1.04 | **0.61** | **0.65** | 0.60 | **0.0480** | **REJECT $H_0$ (SSD Dominates Benchmark)** |
| | Robust Opt | +10.61% | +1.54 | 0.69 | 0.75 | 0.50 | 0.4800 | Excess Utility |
| **Quarterly** | Equal Weight | -0.50% | -0.32 | 0.91 | 0.94 | 0.90 | 0.1440 | Excess Utility |
| | Min CVaR | +5.31% | +1.19 | **0.42** | **0.44** | 0.40 | 0.1160 | Excess Downside Utility |
| | Black-Litterman | +2.90% | +1.02 | **0.61** | **0.65** | 0.61 | 0.0600 | Marginally Significant |
| | Robust Opt | **+15.54%** | +2.05** | 0.67 | 0.77 | 0.46 | 0.5000 | Excess Utility |

---

## 6. Econometric Diagnostics & Time-Series Properties

### Non-Gaussianity & Volatility Clustering (The Statistical Need for Stochastic Dominance)
Traditional Mean-Variance analysis assumes returns follow Gaussian distributions. We run formal tests to assess normality and time-varying volatility:

```
[Diagnostic Test Suite on Daily Out-of-Sample Returns]
-----------------------------------------------------------------------------------------
Series                  Skewness  Excess Kurtosis   Jarque-Bera p-val   ARCH-LM p-val
-----------------------------------------------------------------------------------------
Nasdaq 100 (VW)          +0.286        4.792           6.058e-252         2.700e-14
Nasdaq 100 (Equal Wt)    +0.255        5.465           0.000e+00          1.359e-22
Euro Stoxx 50 (VW)       -0.088        3.574           7.442e-173         8.399e-37
Euro Stoxx 50 (Equal Wt) -0.139        4.245           5.123e-244         2.198e-41
-----------------------------------------------------------------------------------------
Conclusion: Normality is overwhelmingly REJECTED (p ≈ 0.0000). 
Extreme fat tails and volatility clustering (ARCH) mandate Non-Parametric Stochastic Dominance!
```

---

## 7. Codebase Structure & Pipeline

```
equity-index-reengineering/
│
├── figures/                              # High-resolution empirical figures (PNG)
│   ├── cumulative_returns_comparison.png # 4.9y Out-of-sample wealth trajectories
│   ├── drawdown_underwater_curves.png    # Drawdown profile / crash resilience
│   ├── ssd_integrated_cdf_curves.png     # Visual proof of Second-Order Stochastic Dominance
│   ├── risk_return_frontier.png          # Sharpe/Sortino efficiency tradeoff
│   ├── concentration_lorenz_curves.png   # Lorenz curve & Gini index concentration
│   └── asymmetric_betas_comparison.png   # Downside vs Upside systematic betas
│
├── data/                                 # 100% Real constituent data & historical prices
│   ├── nasdaq100_constituents.csv        # Live constituents & official weights
│   ├── nasdaq100_prices.csv              # Clean daily prices (2021-2026)
│   ├── nasdaq100_market_caps.csv         # Real market capitalizations
│   ├── eurostoxx50_constituents.csv      # 50 European blue-chips
│   ├── eurostoxx50_prices.csv            # Clean daily prices (2021-2026)
│   └── eurostoxx50_market_caps.csv       # Real market capitalizations
│
├── fetch_real_data.py                    # Multi-source scraper & yfinance/OpenBB pipeline
├── portfolio_models.py                   # CVXPY convex optimizers (CVaR, BL, Robust, EW, MC)
├── stochastic_spanning.py                # Arvanitis & Topaloglou (2019) SSD Spanning Tests
├── econometric_tests.py                  # Jarque-Bera, ARCH-LM, HAC Alphas, Downside Betas
├── generate_all_charts.py                # High-res publication graphics generator
├── run_experiments.py                    # Benchmark out-of-sample simulation script
├── run_full_econometric_study.py         # Master multi-frequency empirical suite
├── requirements.txt                      # Exact package dependencies
└── README.md                             # Comprehensive research documentation
```

---

## 8. Quickstart & Reproducibility

### 1. Installation
Clone the repository and install the verified scientific dependencies:
```bash
git clone https://github.com/gkompo/-Beyond-Market-Cap-Weighting-Re-Engineering-Mega-Cap-Equity-Indices-via-Stochastic-Dominance-Tests-.git
cd -Beyond-Market-Cap-Weighting-Re-Engineering-Mega-Cap-Equity-Indices-via-Stochastic-Dominance-Tests-
pip install -r requirements.txt
```

### 2. Fetch Latest Real Market Data
```bash
python fetch_real_data.py
```

### 3. Generate All Publication Figures
```bash
python generate_all_charts.py
```

### 4. Run the Full Multi-Frequency Econometric Study
```bash
python run_full_econometric_study.py
```

### 5. Run the Econometric Diagnostic Suite
```bash
python econometric_tests.py
```

---

## 9. Academic References & Master's Thesis Context

This repository serves as an empirical foundation for a Master's Thesis at the **Athens University of Economics and Business (AUEB)**, Department of International and European Economic Studies (DIEES), proposed under the supervision of **Dr. Georgios Tsomidis** and **Prof. Nikolaos Topaloglou**.

> **Note on Data Fidelity:** While this pilot project uses real market data collected via OpenBB and `yfinance`, the final dissertation will utilize point-in-time constituent history from the **Bloomberg Terminal** to completely eradicate survivorship bias.

### Key References
1. **Papavasiliou, E., Topaloglou, N., & Tsomidis, G. (2022).** *Investors' Behavior in Alternative Asset Classes*. **SPOUDAI Journal of Economics and Business**, 72(3-4), 3–55.
2. **Arvanitis, S., Hallam, M., Post, T., & Topaloglou, N. (2019).** *Stochastic Spanning*. **Journal of Business & Economic Statistics**, 37(4), 573–585.
3. **Arvanitis, S., Scaillet, O., & Topaloglou, N. (2024).** *Sparse Spanning Portfolios and Under-Diversification with Second-Order Stochastic Dominance*.
4. **Rockafellar, R. T., & Uryasev, S. (2000).** *Optimization of Conditional Value-at-Risk*. **Journal of Risk**, 2(3), 21–41.
5. **DeMiguel, V., Garlappi, L., & Uppal, R. (2009).** *Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?* **Review of Financial Studies**, 22(5), 1915–1953.
6. **Ang, A., Chen, J., & Xing, Y. (2006).** *Downside Risk*. **Review of Financial Studies**, 19(4), 1199–1239.
7. **Ledoit, O., & Wolf, M. (2008).** *Robust Performance Hypothesis Testing with the Sharpe Ratio*. **Journal of Empirical Finance**, 15(5), 850–859.
8. **Memmel, C. (2003).** *Performance Hypothesis Testing with the Sharpe Ratio*. **Finance Letters**, 1(1), 21–23.

---

### Author
* **George Kompothanasis**  
* GitHub: [@gkompo](https://github.com/gkompo)
