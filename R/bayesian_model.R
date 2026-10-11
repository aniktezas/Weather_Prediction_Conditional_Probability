# =========================================================================
# BAMAT207: Probability and Statistics
# Module: Gaussian Bayesian Classifier from Mathematical First Principles
# =========================================================================

fit_gaussian_bayes <- function(train_df, feature_cols, target_col = "RainTomorrow", eps = 1e-6) {
  y <- train_df[[target_col]]
  n <- nrow(train_df)
  
  classes <- c(0, 1)
  priors <- c(`0` = mean(y == 0), `1` = mean(y == 1))
  
  means <- list()
  variances <- list()
  
  for (c in classes) {
    sub_df <- train_df[y == c, feature_cols]
    means[[as.character(c)]] <- colMeans(sub_df)
    variances[[as.character(c)]] <- apply(sub_df, 2, var) + eps
  }
  
  list(
    feature_cols = feature_cols,
    priors = priors,
    means = means,
    variances = variances
  )
}

predict_gaussian_bayes_prob <- function(model, test_df) {
  X <- test_df[, model$feature_cols]
  n <- nrow(X)
  probs <- numeric(n)
  
  for (i in 1:n) {
    x_i <- as.numeric(X[i, ])
    log_post <- numeric(2)
    
    for (c_idx in 1:2) {
      c_val <- c_idx - 1
      c_str <- as.character(c_val)
      log_prior <- log(model$priors[c_str])
      
      mu <- model$means[[c_str]]
      sigma2 <- model$variances[[c_str]]
      
      # Gaussian log likelihood: -0.5 * log(2*pi*sigma2) - ((x - mu)^2)/(2*sigma2)
      log_lik <- sum(-0.5 * log(2 * pi * sigma2) - ((x_i - mu)^2) / (2 * sigma2))
      log_post[c_idx] <- log_prior + log_lik
    }
    
    # Log-sum-exp normalization
    max_log <- max(log_post)
    exp_terms <- exp(log_post - max_log)
    p_norm <- exp_terms / sum(exp_terms)
    
    probs[i] <- p_norm[2] # P(Y = 1 | X)
  }
  return(probs)
}
