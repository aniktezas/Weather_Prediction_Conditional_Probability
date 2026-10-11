# =========================================================================
# BAMAT207: Probability and Statistics
# Module: Discrete-Time Markov State Transition Models
# =========================================================================

fit_markov_model <- function(train_df, alpha = 1.0) {
  rain_today <- train_df$RainToday
  rain_tomorrow <- train_df$RainTomorrow
  
  # First-Order State Transition Counts: P(W_{t+1}=j | W_t=i)
  # States: 0 = Dry, 1 = Rain
  counts <- matrix(0, nrow = 2, ncol = 2,
                   dimnames = list(c("Dry_t", "Rain_t"), c("Dry_t+1", "Rain_t+1")))
  
  for (i in 1:length(rain_today)) {
    curr <- rain_today[i] + 1
    nxt <- rain_tomorrow[i] + 1
    counts[curr, nxt] <- counts[curr, nxt] + 1
  }
  
  counts_smoothed <- counts + alpha
  transition_matrix <- counts_smoothed / rowSums(counts_smoothed)
  
  # Analytical Stationary Distribution pi = pi * P
  # pi_Dry * P_01 = pi_Rain * P_10
  p01 <- transition_matrix[1, 2]
  p10 <- transition_matrix[2, 1]
  pi_rain <- p01 / (p01 + p10)
  pi_dry <- 1.0 - pi_rain
  stationary_dist <- c(Dry = pi_dry, Rain = pi_rain)
  
  list(
    counts = counts,
    transition_matrix = transition_matrix,
    stationary_dist = stationary_dist
  )
}

predict_markov_prob <- function(model, test_df) {
  # Given RainToday state, return P(RainTomorrow = 1 | RainToday)
  probs <- numeric(nrow(test_df))
  for (i in 1:nrow(test_df)) {
    state <- test_df$RainToday[i] + 1 # R uses 1-based indexing
    probs[i] <- model$transition_matrix[state, 2] # Col 2 is Rain
  }
  return(probs)
}
