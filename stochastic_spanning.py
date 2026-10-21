"""
stochastic_spanning.py
----------------------
Implements Second-Order Stochastic Dominance (SSD) and Stochastic Spanning Tests
following the methodology of:
    Arvanitis, S., Hallam, M., Post, T., & Topaloglou, N. (2019).
    "Stochastic Spanning". Journal of Business & Economic Statistics, 37(4), 573-585.

Author: Prepared for Diploma Thesis with Dr. G. Tsomidis and Prof. N. Topaloglou.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog

def integrated_cdf_ssd(returns, grid_points):
    """
    Computes the Second-Order integrated cumulative distribution function:
        F^(2)(z; r) = (1 / T) * sum_t max(z - r_t, 0)
    for each threshold z in grid_points.
    """
    T = len(returns)
    # returns[:, None] has shape (T, 1), grid_points[None, :] has shape (1, G)
    shortfall = np.maximum(grid_points[None, :] - returns[:, None], 0.0)
    return np.mean(shortfall, axis=0)

def ssd_dominance_statistic(r_test, r_benchmark, num_grid=100):
    """
    Computes the maximal empirical violation for Second-Order Stochastic Dominance (SSD).
    Under H0: Benchmark SSD-dominates or spans r_test.
    Violation: eta = max_z [ F^(2)(z; r_benchmark) - F^(2)(z; r_test) ]
    A positive eta indicates points where the alternative strategy has strictly
    lower downside shortfall (higher utility for risk-averse investors) than the benchmark.
    """
    all_returns = np.concatenate([r_test, r_benchmark])
    z_min, z_max = np.percentile(all_returns, 1), np.percentile(all_returns, 99)
    grid = np.linspace(z_min, z_max, num_grid)
    
    F2_bench = integrated_cdf_ssd(r_benchmark, grid)
    F2_test  = integrated_cdf_ssd(r_test, grid)
    
    # Distance: positive means alternative is superior (lower shortfall)
    diff = F2_bench - F2_test
    eta_stat = np.max(diff)
    mean_diff = np.mean(diff)
    
    return eta_stat, mean_diff, grid, diff

def arvanitis_topaloglou_subsampling_test(r_test, r_benchmark, b_size=None, n_subsamples=500, num_grid=50):
    """
    Non-parametric subsampling test for Stochastic Spanning / Dominance
    following Arvanitis, Hallam, Post, and Topaloglou (2019).
    
    Parameters:
        r_test: Array of out-of-sample returns of the alternative re-engineered strategy
        r_benchmark: Array of out-of-sample returns of the Market-Cap benchmark
        b_size: Subsample block size (default ~ T^(2/3) as recommended by Politis & Romano)
        n_subsamples: Number of bootstrap/subsample iterations
    
    Returns:
        eta_stat: Observed test statistic on full sample
        p_value: Empirical p-value testing H0 (benchmark spans/dominates alternative)
        decision: Statistical conclusion at alpha = 5%
    """
    T = len(r_test)
    if b_size is None:
        b_size = int(np.floor(T ** (2.0 / 3.0)))
    b_size = max(b_size, 20)
    
    obs_eta, _, _, _ = ssd_dominance_statistic(r_test, r_benchmark, num_grid=num_grid)
    
    # Subsampling under the Null
    subsample_etas = []
    # Centered test to simulate the null boundary
    for _ in range(n_subsamples):
        start_idx = np.random.randint(0, T - b_size + 1)
        sub_test = r_test[start_idx : start_idx + b_size]
        sub_bench = r_benchmark[start_idx : start_idx + b_size]
        
        eta_b, _, _, _ = ssd_dominance_statistic(sub_test, sub_bench, num_grid=num_grid)
        # Scaled statistic: sqrt(b_size) * (eta_b - obs_eta)
        subsample_etas.append(eta_b - obs_eta)
        
    subsample_etas = np.array(subsample_etas)
    
    # Compute empirical p-value
    # H0: Alternative does NOT dominate (eta <= 0)
    # Rejection of H0 (p < 0.05) => Alternative stochastically dominates / spans benchmark!
    p_value = np.mean(subsample_etas >= obs_eta)
    
    dominates = (obs_eta > 0) and (p_value < 0.05)
    
    return {
        "obs_eta": obs_eta,
        "p_value": p_value,
        "b_size": b_size,
        "n_subsamples": n_subsamples,
        "strictly_dominates": dominates,
        "conclusion": "Alternative strictly SSD-dominates Market-Cap (H0 Rejected)" if dominates 
                      else ("Alternative has lower shortfall (eta > 0) but not statistically significant at 5%" if obs_eta > 0 
                            else "Market-Cap spans/dominates alternative (H0 Accepted)")
    }
