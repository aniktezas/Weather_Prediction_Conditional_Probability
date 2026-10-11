# =========================================================================
# BAMAT207: Probability and Statistics
# Module: Viterbi Algorithm for 3-State Synoptic Weather Regimes
# =========================================================================

viterbi_decode_r <- function(obs_humidity, obs_pressure) {
  T_steps <- length(obs_humidity)
  n_states <- 3
  
  pi_init <- c(0.50, 0.30, 0.20)
  A <- matrix(c(0.85, 0.12, 0.03,
                0.20, 0.65, 0.15,
                0.08, 0.22, 0.70), nrow = 3, byrow = TRUE)
                
  means_h <- c(48.0, 72.0, 88.0)
  vars_h <- c(80.0, 60.0, 45.0)
  means_p <- c(1013.0, 1008.0, 1003.0)
  vars_p <- c(12.0, 10.0, 15.0)
  
  V <- matrix(0, nrow = T_steps, ncol = n_states)
  backpointer <- matrix(0, nrow = T_steps, ncol = n_states)
  
  # Init
  for (s in 1:n_states) {
    emit <- -0.5 * log(2 * pi * vars_h[s]) - ((obs_humidity[1] - means_h[s])^2) / (2 * vars_h[s]) +
            -0.5 * log(2 * pi * vars_p[s]) - ((obs_pressure[1] - means_p[s])^2) / (2 * vars_p[s])
    V[1, s] <- log(pi_init[s]) + emit
  }
  
  # Forward induction
  for (t in 2:T_steps) {
    for (s in 1:n_states) {
      emit <- -0.5 * log(2 * pi * vars_h[s]) - ((obs_humidity[t] - means_h[s])^2) / (2 * vars_h[s]) +
              -0.5 * log(2 * pi * vars_p[s]) - ((obs_pressure[t] - means_p[s])^2) / (2 * vars_p[s])
      trans_v <- V[t - 1, ] + log(A[, s])
      best_prev <- which.max(trans_v)
      V[t, s] <- trans_v[best_prev] + emit
      backpointer[t, s] <- best_prev
    }
  }
  
  # Backtracking
  best_path <- numeric(T_steps)
  best_path[T_steps] <- which.max(V[T_steps, ])
  for (t in (T_steps - 1):1) {
    best_path[t] <- backpointer[t + 1, best_path[t + 1]]
  }
  
  return(best_path - 1) # 0: Anticyclonic, 1: Convective, 2: Frontal
}
