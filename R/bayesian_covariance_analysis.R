# =========================================================================
# BAMAT207: Probability and Statistics
# Module: Quadratic Discriminant Analysis (Full Covariance) & Dirichlet Updating
# =========================================================================

fit_qda_r <- function(train_df, feature_cols, target_col = "RainTomorrow", reg = 1e-4) {
  y <- train_df[[target_col]]
  n <- nrow(train_df)
  p <- length(feature_cols)
  
  classes <- c(0, 1)
  priors <- c(`0` = mean(y == 0), `1` = mean(y == 1))
  
  means <- list()
  inv_covs <- list()
  log_det_covs <- list()
  
  for (c in classes) {
    X_c <- as.matrix(train_df[y == c, feature_cols])
    mu_c <- colMeans(X_c)
    means[[as.character(c)]] <- mu_c
    
    # Full covariance matrix with ridge stabilization
    cov_c <- cov(X_c) + diag(reg, p)
    inv_covs[[as.character(c)]] <- solve(cov_c)
    log_det_covs[[as.character(c)]] <- as.numeric(determinant(cov_c, logarithm = TRUE)$modulus)
  }
  
  list(
    feature_cols = feature_cols,
    priors = priors,
    means = means,
    inv_covs = inv_covs,
    log_det_covs = log_det_covs
  )
}

predict_qda_prob_r <- function(model, test_df) {
  X <- as.matrix(test_df[, model$feature_cols])
  n <- nrow(X)
  probs <- numeric(n)
  p <- length(model$feature_cols)
  
  for (i in 1:n) {
    x_i <- X[i, ]
    log_post <- numeric(2)
    
    for (c_idx in 1:2) {
      c_str <- as.character(c_idx - 1)
      diff <- x_i - model$means[[c_str]]
      maha <- as.numeric(t(diff) %*% model$inv_covs[[c_str]] %*% diff)
      log_lik <- -0.5 * (model$log_det_covs[[c_str]] + maha + p * log(2 * pi))
      log_post[c_idx] <- log(model$priors[c_str]) + log_lik
    }
    
    max_log <- max(log_post)
    exp_terms <- exp(log_post - max_log)
    probs[i] <- exp_terms[2] / sum(exp_terms)
  }
  return(probs)
}
