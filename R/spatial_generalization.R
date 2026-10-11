# =========================================================================
# BAMAT207: Probability and Statistics
# Module: Spatial Cross-Regional Model Generalization (Pune -> Delhi)
# =========================================================================

run_spatial_generalization_r <- function(data_dir) {
  pune_file <- file.path(data_dir, "weather_pune_10yr.csv")
  delhi_file <- file.path(data_dir, "weather_delhi_10yr.csv")
  
  if (!file.exists(pune_file) || !file.exists(delhi_file)) return(NULL)
  
  df_pune <- load_weather_data(pune_file)
  df_delhi <- load_weather_data(delhi_file)
  
  # Train models on Pune
  cond_model <- fit_conditional_model(df_pune)
  feature_cols <- c("MaxTemp", "MinTemp", "TempRange", "Precipitation", "Humidity", "Pressure", "WindSpeed", "RainToday")
  bayes_model <- fit_gaussian_bayes(df_pune, feature_cols = feature_cols)
  
  # Test on Delhi
  p_cond_delhi <- predict_conditional_prob(cond_model, df_delhi)
  p_bayes_delhi <- predict_gaussian_bayes_prob(bayes_model, df_delhi)
  
  m_cond <- compute_metrics_r(df_delhi$RainTomorrow, p_cond_delhi)
  m_bayes <- compute_metrics_r(df_delhi$RainTomorrow, p_bayes_delhi)
  
  list(
    m_cond = m_cond,
    m_bayes = m_bayes
  )
}
