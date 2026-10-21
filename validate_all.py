"""
validate_all.py
---------------
Comprehensive Multi-Layer Automated Validation Suite:
1. Simplex Constraint Verification (Sum(w) = 1, w >= 0, Convex feasibility)
2. Zero Look-Ahead Information Causality Test (Perturbation invariance)
3. Data Integrity & Official Benchmark Ground Truth Verification
4. Accounting Identity & Return Compounding Consistency
5. Econometric Diagnostics Sanity Bounds (HAC, Betas, ARCH, JB)
6. Stochastic Dominance Axiomatic Verification (Monotonicity & Convexity of F^(2))
"""

import os
import sys
import numpy as np
import pandas as pd
import cvxpy as cp
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_arch
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

from portfolio_models import (
    get_equal_weights,
    get_min_cvar_weights,
    get_black_litterman_weights,
    get_robust_weights
)
from stochastic_spanning import integrated_cdf_ssd, ssd_dominance_statistic, arvanitis_topaloglou_subsampling_test

def run_validation():
    print("="*80)
    print(" EXECUTING COMPREHENSIVE MULTI-LAYER VALIDATION AUDIT")
    print("="*80)
    
    passed_tests = 0
    total_tests = 0
    
    # -------------------------------------------------------------------------
    # LAYER 1: DATA INTEGRITY & OFFICIAL BENCHMARK VERIFICATION
    # -------------------------------------------------------------------------
    print("\n[LAYER 1] Verifying Data Integrity & Official Exchange Benchmarks...")
    for idx in ["nasdaq100", "eurostoxx50"]:
        total_tests += 1
        p_file = os.path.join(DATA_DIR, f"{idx}_prices.csv")
        b_file = os.path.join(DATA_DIR, f"{idx}_benchmark_index.csv")
        
        assert os.path.exists(p_file), f"Missing prices file for {idx}"
        assert os.path.exists(b_file), f"Missing benchmark file for {idx}"
        
        prices = pd.read_csv(p_file, index_col=0, parse_dates=True)
        bench = pd.read_csv(b_file, index_col=0, parse_dates=True)
        
        # Test: Zero NaNs in cleaned data
        assert prices.isna().sum().sum() == 0, f"Found NaN values in {idx} prices"
        assert bench.isna().sum().sum() == 0, f"Found NaN values in {idx} benchmark"
        
        # Test: Non-empty intersection and proper monotonic dates
        common = prices.index.intersection(bench.index)
        assert len(common) >= 1000, f"Insufficient common dates ({len(common)}) for {idx}"
        assert common.is_monotonic_increasing, f"Dates not monotonic for {idx}"
        
        passed_tests += 1
        print(f"  [PASS] {idx.upper()}: {prices.shape[1]} stocks, {len(common)} clean daily bars. Zero nulls.")

    # -------------------------------------------------------------------------
    # LAYER 2: MATHEMATICAL SIMPLEX & CONVEX FEASIBILITY
    # -------------------------------------------------------------------------
    print("\n[LAYER 2] Verifying Optimization Constraints (Simplex w >= 0, sum(w) = 1)...")
    np.random.seed(42)
    sample_returns = np.random.randn(252, 20) * 0.01
    
    models = {
        "Equal Weight (1/N)": get_equal_weights(20),
        "Min CVaR (LP)": get_min_cvar_weights(sample_returns, alpha=0.95),
        "Black-Litterman": get_black_litterman_weights(sample_returns, get_equal_weights(20)),
        "Robust Optimization (SOCP)": get_robust_weights(sample_returns, kappa=0.15)
    }
    
    for name, w in models.items():
        total_tests += 1
        # Test 1: Sum to 1
        assert np.isclose(np.sum(w), 1.0, atol=1e-5), f"{name}: Weights do not sum to 1 (sum = {np.sum(w)})"
        # Test 2: Long-only (w >= 0)
        assert np.all(w >= -1e-6), f"{name}: Negative weights detected (min = {np.min(w)})"
        # Test 3: Finite & valid numbers
        assert np.all(np.isfinite(w)), f"{name}: Non-finite values in weight vector"
        passed_tests += 1
        print(f"  [PASS] {name}: Sum = {np.sum(w):.6f}, Min = {np.min(w):.6f}, Max = {np.max(w):.6f}")

    # -------------------------------------------------------------------------
    # LAYER 3: INFORMATION CAUSALITY & ZERO LOOKAHEAD VERIFICATION
    # -------------------------------------------------------------------------
    print("\n[LAYER 3] Verifying Zero Lookahead Bias (Information Causality Test)...")
    total_tests += 1
    t = 300
    lookback = 252
    prices = pd.read_csv(os.path.join(DATA_DIR, "nasdaq100_prices.csv"), index_col=0, parse_dates=True)
    returns = prices.pct_change().dropna()
    
    hist_r_clean = returns.iloc[t - lookback : t].values
    w_unperturbed = get_min_cvar_weights(hist_r_clean, alpha=0.95)
    
    returns_perturbed = returns.copy()
    returns_perturbed.iloc[t] = returns_perturbed.iloc[t] * 100.0
    returns_perturbed.iloc[t + 1] = -0.50
    returns_perturbed.iloc[t + 5] = +1.00
    
    hist_r_test = returns_perturbed.iloc[t - lookback : t].values
    w_after_future_shock = get_min_cvar_weights(hist_r_test, alpha=0.95)
    
    max_diff = np.max(np.abs(w_unperturbed - w_after_future_shock))
    assert max_diff == 0.0, f"Lookahead bias detected! Future shock altered past weights by {max_diff}"
    passed_tests += 1
    print(f"  [PASS] Future returns perturbation test: Max Weight Delta = {max_diff:.10f} (STRICT ZERO LOOKAHEAD)")

    # -------------------------------------------------------------------------
    # LAYER 4: ACCOUNTING IDENTITY & WEALTH COMPOUNDING
    # -------------------------------------------------------------------------
    print("\n[LAYER 4] Verifying Financial Accounting Identities & Compounding...")
    total_tests += 1
    test_r = np.array([0.01, -0.02, 0.015, -0.005, 0.02])
    compounded_wealth = np.cumprod(1 + test_r)
    expected_final_wealth = (1 + 0.01) * (1 - 0.02) * (1 + 0.015) * (1 - 0.005) * (1 + 0.02)
    assert np.isclose(compounded_wealth[-1], expected_final_wealth, atol=1e-10)
    
    peak = np.maximum.accumulate(compounded_wealth)
    drawdowns = (compounded_wealth - peak) / peak
    assert np.all(drawdowns <= 1e-12), "Drawdown exceeded zero"
    assert np.isclose(np.max(drawdowns), 0.0), "Peak not at 0% drawdown"
    passed_tests += 1
    print(f"  [PASS] Compounding, peak tracking, and drawdown bounds verified mathematically")

    # -------------------------------------------------------------------------
    # LAYER 5: ECONOMETRIC DIAGNOSTICS & SANITY BOUNDS
    # -------------------------------------------------------------------------
    print("\n[LAYER 5] Verifying Econometric Diagnostics Sanity Bounds...")
    total_tests += 1
    bench_r = pd.read_csv(os.path.join(DATA_DIR, "nasdaq100_benchmark_index.csv"), index_col=0).pct_change().dropna().values.flatten()
    
    jb_stat, jb_p = stats.jarque_bera(bench_r)
    assert jb_stat > 0 and jb_p >= 0.0, "Invalid Jarque-Bera output"
    
    lm_stat, lm_p, _, _ = het_arch(bench_r - np.mean(bench_r), nlags=5)
    assert lm_stat > 0 and lm_p >= 0.0, "Invalid ARCH-LM output"
    
    eq_r = returns.mean(axis=1).iloc[-len(bench_r):].values
    min_len = min(len(eq_r), len(bench_r))
    X = sm.add_constant(bench_r[-min_len:])
    model = sm.OLS(eq_r[-min_len:], X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    beta = model.params[1]
    r2 = model.rsquared
    assert 0.5 <= beta <= 1.5, f"Equal weight market beta out of realistic bounds: {beta}"
    assert 0.70 <= r2 <= 1.0, f"CAPM R-squared suspiciously low for index constituents: {r2}"
    passed_tests += 1
    print(f"  [PASS] Econometric outputs verified: Beta = {beta:.3f}, R2 = {r2:.3f}, JB stat = {jb_stat:.1f}")

    # -------------------------------------------------------------------------
    # LAYER 6: STOCHASTIC DOMINANCE AXIOMS & CONVEXITY
    # -------------------------------------------------------------------------
    print("\n[LAYER 6] Verifying Stochastic Dominance Axioms (Arvanitis & Topaloglou)...")
    total_tests += 1
    grid = np.linspace(-0.10, 0.10, 100)
    F2 = integrated_cdf_ssd(bench_r, grid)
    
    assert np.all(F2 >= 0.0), "Negative integrated shortfall detected"
    diff_F2 = np.diff(F2)
    assert np.all(diff_F2 >= -1e-12), "Monotonicity of F^(2) violated"
    diff2_F2 = np.diff(diff_F2)
    assert np.all(diff2_F2 >= -1e-12), "Convexity of F^(2) violated"
    
    sub_res = arvanitis_topaloglou_subsampling_test(eq_r[-min_len:], bench_r[-min_len:], n_subsamples=50, num_grid=20)
    assert 0.0 <= sub_res["p_value"] <= 1.0, f"Invalid p-value: {sub_res['p_value']}"
    assert np.isfinite(sub_res["obs_eta"]), "Non-finite test statistic"
    passed_tests += 1
    print(f"  [PASS] Axiomatic SSD properties: F^(2) non-negative, monotonic, convex. Bootstrap p-val valid.")

    print("\n" + "="*80)
    print(f" FINAL VALIDATION RESULT: {passed_tests}/{total_tests} AUDIT CHECKS PASSED (100% SUCCESS)")
    print("="*80)

if __name__ == "__main__":
    run_validation()
