# =========================================================================
# BAMAT207: Probability and Statistics
# Module: Comparative Benchmarks (Climatology & Logistic Regression)
# =========================================================================

fit_climatology_baseline <- function(train_df, target_col = "RainTomorrow") {
  prior <- mean(train_df[[target_col]])
  list(prior = prior)
}

predict_climatology_prob <- function(model, n_samples) {
  rep(model$prior, n_samples)
}

fit_logistic_regression <- function(train_df, formula) {
  glm_model <- glm(formula, data = train_df, family = binomial(link = "logit"))
  return(glm_model)
}

predict_logistic_prob <- function(glm_model, test_df) {
  predict(glm_model, newdata = test_df, type = "response")
}
