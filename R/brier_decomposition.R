# =========================================================================
# BAMAT207: Probability and Statistics
# Module: Murphy (1973) Brier Score Formal Decomposition in R
# BS = Reliability - Resolution + Uncertainty
# =========================================================================

murphy_brier_decomposition_r <- function(y_true, y_probs, n_bins = 10) {
  N <- length(y_true)
  c_base <- mean(y_true)
  uncertainty <- c_base * (1.0 - c_base)
  
  bins <- seq(0.0, 1.0, length.out = n_bins + 1)
  reliability <- 0.0
  resolution <- 0.0
  
  for (k in 1:n_bins) {
    low <- bins[k]
    high <- bins[k + 1]
    mask <- if (k == n_bins) (y_probs >= low & y_probs <= high) else (y_probs >= low & y_probs < high)
    n_k <- sum(mask)
    if (n_k > 0) {
      p_k_bar <- mean(y_probs[mask])
      y_k_bar <- mean(y_true[mask])
      rel_k <- (n_k / N) * ((p_k_bar - y_k_bar)^2)
      res_k <- (n_k / N) * ((y_k_bar - c_base)^2)
      reliability <- reliability + rel_k
      resolution <- resolution + res_k
    }
  }
  
  raw_brier <- mean((y_probs - y_true)^2)
  decomposed_brier <- reliability - resolution + uncertainty
  bss <- 1.0 - (raw_brier / uncertainty)
  
  list(
    raw_brier = round(raw_brier, 4),
    decomposed_brier = round(decomposed_brier, 4),
    reliability = round(reliability, 4),
    resolution = round(resolution, 4),
    uncertainty = round(uncertainty, 4),
    brier_skill_score = round(bss, 4)
  )
}
