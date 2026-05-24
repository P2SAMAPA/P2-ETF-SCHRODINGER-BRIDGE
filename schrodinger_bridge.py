import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

def sinkhorn(a, b, K, reg_eps, max_iter=100, stop_thresh=1e-6):
    """Sinkhorn algorithm for entropic OT."""
    v = np.ones(len(b))
    for _ in range(max_iter):
        u = a / (K @ v)
        v_new = b / (K.T @ u)
        err = np.linalg.norm(v_new - v)
        v = v_new
        if err < stop_thresh:
            break
    # Coupling matrix
    Pi = np.diag(u) @ K @ np.diag(v)
    return Pi

def compute_schrodinger_scores(returns, reg_eps=0.01, max_iter=100, stop_thresh=1e-6):
    """
    Build Schrödinger bridge between start and end of the window.
    Use returns of the first and last day as two distributions over assets.
    Returns per-ETF score: expected relative displacement or mass change.
    """
    returns_clean = returns.dropna()
    if returns_clean.shape[1] < 2:
        return {t: 0.0 for t in returns_clean.columns}
    # Take first and last day returns as source and target distributions (over assets)
    source = returns_clean.iloc[0].values
    target = returns_clean.iloc[-1].values
    # Standardize to sum to 1 (make them probability distributions)
    source = np.abs(source) / np.sum(np.abs(source))
    target = np.abs(target) / np.sum(np.abs(target))
    # Compute cost matrix: Euclidean distance between assets' historical returns? Here we use return values as features.
    # Better: use the full time series of returns for each asset as feature vectors.
    # To keep simple, we compute distance between the average return profiles.
    features = returns_clean.T.values  # shape (n_assets, T)
    cost = cdist(features, features, metric='euclidean')
    # Kernel matrix: exp(-cost / reg_eps)
    K = np.exp(-cost / reg_eps)
    # Compute coupling
    Pi = sinkhorn(source, target, K, reg_eps, max_iter, stop_thresh)
    # Per-asset score: expected log displacement in the bridge (Kullback‑Leibler divergence of row sums)
    # Or simply: the sum of probability mass transported from asset i (source mass * row sum)
    # Higher score means the asset is a source of mass that flows to many others.
    # We'll use the row sum of Pi (mass sent from i) as score.
    row_sum = Pi.sum(axis=1)
    tickers = returns_clean.columns
    return {ticker: row_sum[i] for i, ticker in enumerate(tickers)}
