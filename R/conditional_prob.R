# =========================================================================
# BAMAT207: Probability and Statistics
# Module: Empirical Conditional Probability & Law of Total Probability
# =========================================================================

fit_conditional_model <- function(train_df, alpha = 1.0) {
  y <- train_df$RainTomorrow
  n_total <- length(y)
  
  # 1. Prior Probability P(Rain)
  prior_rain <- mean(y)
  prior_dry <- 1 - prior_rain
  
  # 2. 1D Conditional Probabilities P(Rain=1 | Humidity=h)
  hum_cats <- levels(train_df$Humidity_Cat)
  p_rain_given_hum <- numeric(length(hum_cats))
  names(p_rain_given_hum) <- hum_cats
  
  for (h in hum_cats) {
    mask <- (train_df$Humidity_Cat == h)
    cnt_h <- sum(mask)
    cnt_rain_h <- sum(y[mask] == 1)
    # Laplace smoothing
    p_rain_given_hum[h] <- (cnt_rain_h + alpha) / (cnt_h + 2 * alpha)
  }
  
  # 3. 2D Multi-Condition Probability P(Rain=1 | Humidity=h, Pressure=p)
  press_cats <- levels(train_df$Pressure_Cat)
  cond_matrix <- matrix(0, nrow = length(hum_cats), ncol = length(press_cats),
                        dimnames = list(hum_cats, press_cats))
  
  for (h in hum_cats) {
    for (p in press_cats) {
      mask <- (train_df$Humidity_Cat == h) & (train_df$Pressure_Cat == p)
      cnt_hp <- sum(mask)
      cnt_rain_hp <- sum(y[mask] == 1)
      cond_matrix[h, p] <- (cnt_rain_hp + alpha) / (cnt_hp + 2 * alpha)
    }
  }
  
  model <- list(
    alpha = alpha,
    prior_rain = prior_rain,
    prior_dry = prior_dry,
    p_rain_given_hum = p_rain_given_hum,
    cond_matrix = cond_matrix
  )
  return(model)
}

verify_law_of_total_probability <- function(model, train_df) {
  hum_table <- table(train_df$Humidity_Cat)
  p_hum <- as.numeric(prop.table(hum_table))
  names(p_hum) <- names(hum_table)
  
  reconstructed_p_rain <- 0.0
  details <- list()
  
  for (h in names(p_hum)) {
    p_h <- p_hum[h]
    p_r_given_h <- model$p_rain_given_hum[h]
    contrib <- p_h * p_r_given_h
    reconstructed_p_rain <- reconstructed_p_rain + contrib
    details[[h]] <- c(P_H = p_h, P_Rain_given_H = p_r_given_h, Contribution = contrib)
  }
  
  delta <- abs(model$prior_rain - reconstructed_p_rain)
  
  list(
    direct_prior = model$prior_rain,
    reconstructed = reconstructed_p_rain,
    delta = delta,
    partition_details = do.call(rbind, details)
  )
}

predict_conditional_prob <- function(model, test_df) {
  n <- nrow(test_df)
  probs <- numeric(n)
  
  for (i in 1:n) {
    h <- as.character(test_df$Humidity_Cat[i])
    p <- as.character(test_df$Pressure_Cat[i])
    if (h %in% rownames(model$cond_matrix) && p %in% colnames(model$cond_matrix)) {
      probs[i] <- model$cond_matrix[h, p]
    } else {
      probs[i] <- model$prior_rain
    }
  }
  return(probs)
}
