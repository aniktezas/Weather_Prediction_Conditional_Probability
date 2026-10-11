import numpy as np

def murphy_brier_decomposition(y_true, y_probs, n_bins=10):
    """
    Murphy's (1973) Formal Decomposition of the Brier Score:
    BS = Reliability - Resolution + Uncertainty
    
    Where:
    - Uncertainty = c * (1 - c), with c = mean(y_true)
    - Reliability = sum_k (n_k / N) * (p_k_bar - y_k_bar)^2
    - Resolution  = sum_k (n_k / N) * (y_k_bar - c)^2
    """
    N = len(y_true)
    c = np.mean(y_true)
    uncertainty = c * (1.0 - c)
    
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    reliability = 0.0
    resolution = 0.0
    bin_details = []
    
    for k in range(n_bins):
        low, high = bins[k], bins[k + 1]
        mask = (y_probs >= low) & (y_probs <= high) if k == n_bins - 1 else (y_probs >= low) & (y_probs < high)
        n_k = np.sum(mask)
        if n_k > 0:
            p_k_bar = np.mean(y_probs[mask])
            y_k_bar = np.mean(y_true[mask])
            rel_k = (n_k / N) * ((p_k_bar - y_k_bar) ** 2)
            res_k = (n_k / N) * ((y_k_bar - c) ** 2)
            reliability += rel_k
            resolution += res_k
            bin_details.append({
                "bin": f"[{low:.2f}, {high:.2f}]",
                "count": int(n_k),
                "mean_pred": float(p_k_bar),
                "obs_freq": float(y_k_bar)
            })
            
    raw_brier = np.mean((y_probs - y_true) ** 2)
    decomposed_brier = reliability - resolution + uncertainty
    
    return {
        "raw_brier": float(raw_brier),
        "decomposed_brier": float(decomposed_brier),
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "brier_skill_score": float(1.0 - (raw_brier / (uncertainty + 1e-12))),
        "bin_details": bin_details
    }
