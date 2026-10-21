"""
portfolio_models.py
-------------------
Implements index re-engineering weighting models:
1. Market-Cap Weighted (Benchmark)
2. Equal Weight (1/N)
3. Minimum CVaR (Rockafellar & Uryasev scenario-based downside risk)
4. Black-Litterman (Equilibrium implied returns + Shrinkage)
5. Worst-Case Robust Portfolio Optimization (Ellipsoidal uncertainty)

All models operate on the EXACT SAME constituent universe (w_i >= 0, sum(w_i) = 1).
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize

def get_market_cap_weights(market_caps_series):
    """Normalized market capitalization weights."""
    mc = market_caps_series.fillna(market_caps_series.median())
    mc = np.maximum(mc, 1e-6)
    w = mc / mc.sum()
    return w.values

def get_equal_weights(n_assets):
    """1/N naive diversification weights."""
    return np.ones(n_assets) / n_assets

def get_min_cvar_weights(returns_matrix, alpha=0.95):
    """
    Scenario-based Downside Risk / Minimum CVaR (Rockafellar & Uryasev 2000).
    Formulated as a convex linear program (LP):
        min  gamma + 1 / ((1 - alpha) * T) * sum(u_t)
        s.t. u_t >= - r_t^T w - gamma
             u_t >= 0
             sum(w) = 1
             w >= 0
    """
    T, N = returns_matrix.shape
    w = cp.Variable(N)
    gamma = cp.Variable()
    u = cp.Variable(T)
    
    portfolio_losses = -returns_matrix @ w
    objective = cp.Minimize(gamma + (1.0 / ((1.0 - alpha) * T)) * cp.sum(u))
    constraints = [
        u >= portfolio_losses - gamma,
        u >= 0,
        cp.sum(w) == 1,
        w >= 0
    ]
    
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL)
        if prob.status in ['optimal', 'optimal_inaccurate'] and w.value is not None:
            weights = np.maximum(w.value, 0.0)
            return weights / weights.sum()
    except Exception:
        pass
    return get_equal_weights(N)

def get_black_litterman_weights(returns_matrix, mc_weights, delta=2.5, tau=0.05):
    """
    Black-Litterman Model:
    Reverse-engineers implied equilibrium returns from market-cap weights:
        Pi = delta * Sigma * w_mc
    Blended with shrinkage view towards equal weights.
    """
    N = returns_matrix.shape[1]
    Sigma = np.cov(returns_matrix, rowvar=False) + 1e-5 * np.eye(N)
    
    # Implied equilibrium excess returns
    Pi = delta * Sigma @ mc_weights
    
    # Prior mean and covariance
    mu_prior = Pi
    Sigma_prior = tau * Sigma
    
    # Views: shrinkage towards cross-sectional mean
    P = np.eye(N)
    Q = np.mean(Pi) * np.ones(N)
    Omega = np.diag(np.diag(tau * Sigma))
    
    # BL posterior distribution:
    inv_Sigma_prior = np.linalg.pinv(Sigma_prior)
    inv_Omega = np.linalg.pinv(Omega)
    
    post_cov = np.linalg.pinv(inv_Sigma_prior + P.T @ inv_Omega @ P)
    post_mu = post_cov @ (inv_Sigma_prior @ mu_prior + P.T @ inv_Omega @ Q)
    
    # Solve Markowitz utility: max mu^T w - (delta/2) w^T Sigma w
    w = cp.Variable(N)
    objective = cp.Maximize(post_mu @ w - (delta / 2.0) * cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL)
        if prob.status in ['optimal', 'optimal_inaccurate'] and w.value is not None:
            weights = np.maximum(w.value, 0.0)
            return weights / weights.sum()
    except Exception:
        pass
    return mc_weights

def get_robust_weights(returns_matrix, kappa=0.2, risk_aversion=2.0):
    """
    Worst-case Robust Portfolio Optimization under Ellipsoidal Uncertainty:
        max_w  (mu^T w - kappa * ||Sigma^{1/2} w||_2 - (lambda/2) * w^T Sigma w)
        s.t.   sum(w) == 1, w >= 0
    Protects against estimation risk in asset mean returns.
    """
    N = returns_matrix.shape[1]
    mu = np.mean(returns_matrix, axis=0)
    Sigma = np.cov(returns_matrix, rowvar=False) + 1e-5 * np.eye(N)
    
    w = cp.Variable(N)
    # Norm term represents worst-case uncertainty bound
    uncertainty_penalty = kappa * cp.norm(np.linalg.cholesky(Sigma).T @ w, 2)
    risk_penalty = (risk_aversion / 2.0) * cp.quad_form(w, Sigma)
    
    objective = cp.Maximize(mu @ w - uncertainty_penalty - risk_penalty)
    constraints = [cp.sum(w) == 1, w >= 0]
    
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL)
        if prob.status in ['optimal', 'optimal_inaccurate'] and w.value is not None:
            weights = np.maximum(w.value, 0.0)
            return weights / weights.sum()
    except Exception:
        pass
    return get_equal_weights(N)
