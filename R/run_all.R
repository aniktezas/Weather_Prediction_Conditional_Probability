# =========================================================================
# BAMAT207: Probability and Statistics
# Course Project: Weather Prediction using Conditional Probability
# Faculty Evaluator: Dr. Pallavi Mishra
# Track 1: Bayesian Intelligence
# Master R Script
# =========================================================================

# Get script directory
args <- commandArgs(trailingOnly = FALSE)
script_name <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_name) > 0) {
  base_dir <- dirname(normalizePath(script_name))
} else {
  base_dir <- getwd()
}

project_root <- normalizePath(file.path(base_dir, ".."))
cat("===============================================================================\n")
cat(" BAMAT207: Probability and Statistics Course Project (R Implementation)\n")
cat(" Track 1: Bayesian Intelligence | Weather Prediction using Conditional Probability\n")
cat(" Faculty Evaluator: Dr. Pallavi Mishra\n")
cat("===============================================================================\n")

# Source all modular R scripts
source(file.path(base_dir, "data_loader.R"))
source(file.path(base_dir, "conditional_prob.R"))
source(file.path(base_dir, "markov_model.R"))
source(file.path(base_dir, "bayesian_model.R"))
source(file.path(base_dir, "comparative_models.R"))
source(file.path(base_dir, "evaluation.R"))
source(file.path(base_dir, "visualizer.R"))
source(file.path(base_dir, "brier_decomposition.R"))
source(file.path(base_dir, "bayesian_covariance_analysis.R"))
source(file.path(base_dir, "advanced_markov_hmm.R"))
source(file.path(base_dir, "spatial_generalization.R"))

# 1. Load Data
csv_path <- file.path(project_root, "data", "weather_data.csv")
df <- load_weather_data(csv_path)
splits <- train_test_split_temporal(df, train_ratio = 0.8)
train_df <- splits$train
test_df <- splits$test

# 2. Model 1: Empirical Conditional Probability & Law of Total Probability
cat("\n[1/6] Training Empirical Conditional Probability Model in R...\n")
cond_model <- fit_conditional_model(train_df, alpha = 1.0)
tot_prob <- verify_law_of_total_probability(cond_model, train_df)

cat(sprintf("  -> Law of Total Probability Verification:\n"))
cat(sprintf("     Direct Climatological Prior P(Rain): %.4f\n", tot_prob$direct_prior))
cat(sprintf("     Reconstructed sum_h P(Rain|H)*P(H):  %.4f\n", tot_prob$reconstructed))
cat(sprintf("     Absolute Error (Identity Check):     %.6f\n", tot_prob$delta))

# 3. Model 2: Discrete Markov Chain
cat("\n[2/6] Training Discrete Markov Transition Model in R...\n")
markov_model <- fit_markov_model(train_df, alpha = 1.0)
T_mat <- markov_model$transition_matrix
cat(sprintf("  -> Transition Matrix P(W_{t+1} | W_t):\n"))
cat(sprintf("     P(Dry->Dry)  = %.4f | P(Dry->Rain)  = %.4f\n", T_mat[1,1], T_mat[1,2]))
cat(sprintf("     P(Rain->Dry) = %.4f | P(Rain->Rain) = %.4f\n", T_mat[2,1], T_mat[2,2]))
cat(sprintf("  -> Markov Stationary Distribution [pi_Dry, pi_Rain]: [%.4f, %.4f]\n",
            markov_model$stationary_dist[1], markov_model$stationary_dist[2]))

# 4. Model 3: Continuous Gaussian Bayesian Classifier from Scratch
cat("\n[3/6] Training Continuous Gaussian Bayesian Classifier in R...\n")
feature_cols <- c("MaxTemp", "MinTemp", "TempRange", "Precipitation", "Humidity", "Pressure", "WindSpeed", "RainToday")
bayes_model <- fit_gaussian_bayes(train_df, feature_cols = feature_cols)

# 5. Model 4: Full-Covariance QDA in R
cat("\n[4/6] Training Full-Covariance Quadratic Discriminant Analysis in R...\n")
qda_model <- fit_qda_r(train_df, feature_cols = feature_cols)

# 6. Model 5: Climatology Baseline & Logistic Regression
cat("\n[5/6] Training Benchmarks (Climatology Prior & GLM Logistic Regression)...\n")
clim_model <- fit_climatology_baseline(train_df)
glm_model <- fit_logistic_regression(train_df, formula = RainTomorrow ~ MaxTemp + MinTemp + TempRange + Precipitation + Humidity + Pressure + WindSpeed + RainToday)

# 7. Inference on 731 Unseen Test Days
cat("\n[6/6] Evaluating Predictions on 731-Day Unseen Test Horizon...\n")
y_test <- test_df$RainTomorrow

test_probs <- list(
  `Climatology Baseline` = predict_climatology_prob(clim_model, length(y_test)),
  `Markov Chain (1st Order)` = predict_markov_prob(markov_model, test_df),
  `Empirical Conditional Prob` = predict_conditional_prob(cond_model, test_df),
  `Gaussian Naive Bayes` = predict_gaussian_bayes_prob(bayes_model, test_df),
  `Full Covariance QDA` = predict_qda_prob_r(qda_model, test_df),
  `Logistic Regression` = predict_logistic_prob(glm_model, test_df)
)

metrics_list <- list()
cat("\n==========================================================================================\n")
cat(sprintf("%-28s | %-9s | %-9s | %-9s | %-9s | %-9s | %-9s\n",
            "Model (R Implementation)", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Brier"))
cat("==========================================================================================\n")

for (name in names(test_probs)) {
  m <- compute_metrics_r(y_test, test_probs[[name]], threshold = 0.5)
  metrics_list[[name]] <- m
  cat(sprintf("%-28s | %-9.4f | %-9.4f | %-9.4f | %-9.4f | %-9.4f | %-9.4f\n",
              name, m$Accuracy, m$Precision, m$Recall, m$F1, m$ROC_AUC, m$Brier_Score))
}
cat("==========================================================================================\n")

# Murphy's Brier Score Decomposition in R
brier_decomp_r <- murphy_brier_decomposition_r(y_test, test_probs$`Gaussian Naive Bayes`)
cat("\n--- R: Murphy (1973) Brier Score Formal Decomposition (Gaussian Naive Bayes) ---\n")
cat(sprintf("  Uncertainty (Climatological Difficulty): %.4f\n", brier_decomp_r$uncertainty))
cat(sprintf("  Reliability (Calibration Error, lower is better): %.4f\n", brier_decomp_r$reliability))
cat(sprintf("  Resolution  (Discrimination Power, higher is better): %.4f\n", brier_decomp_r$resolution))
cat(sprintf("  Decomposed Brier = Rel - Res + Unc: %.4f\n", brier_decomp_r$decomposed_brier))
cat(sprintf("  Brier Skill Score (BSS vs Climatology): %.4f\n", brier_decomp_r$brier_skill_score))

# HMM Viterbi Weather Regimes in R
viterbi_states_r <- viterbi_decode_r(test_df$Humidity, test_df$Pressure)
v_counts <- table(factor(viterbi_states_r, levels = 0:2, labels = c("0:Anticyclonic", "1:Convective", "2:Frontal")))
cat(sprintf("\n--- R: HMM Viterbi Latent Atmospheric Regimes Decoded (Test Days) ---\n"))
print(v_counts)

# Spatial Generalization Test in R
spatial_r <- run_spatial_generalization_r(file.path(project_root, "data"))
if (!is.null(spatial_r)) {
  cat(sprintf("\n--- R: Spatial Generalization Test (Trained on Pune -> Evaluated on Delhi) ---\n"))
  cat(sprintf("  Empirical Conditional Probability on Delhi -> Accuracy: %.4f, ROC-AUC: %.4f\n",
              spatial_r$m_cond$Accuracy, spatial_r$m_cond$ROC_AUC))
  cat(sprintf("  Gaussian Naive Bayes on Delhi              -> Accuracy: %.4f, ROC-AUC: %.4f\n",
              spatial_r$m_bayes$Accuracy, spatial_r$m_bayes$ROC_AUC))
}

# Save Metrics CSV
metrics_df <- do.call(rbind, lapply(names(metrics_list), function(n) {
  m <- metrics_list[[n]]
  data.frame(Model = n, Accuracy = m$Accuracy, Precision = m$Precision, Recall = m$Recall,
             F1_Score = m$F1, ROC_AUC = m$ROC_AUC, Brier_Score = m$Brier_Score, Log_Loss = m$Log_Loss)
}))
r_results_dir <- file.path(project_root, "R_results")
dir.create(r_results_dir, showWarnings = FALSE, recursive = TRUE)
write.csv(metrics_df, file.path(r_results_dir, "r_metrics_summary.csv"), row.names = FALSE)
cat(sprintf("\n[R Output] Metrics saved to: %s\n", file.path(r_results_dir, "r_metrics_summary.csv")))

# Generate Visualizations
fig_dir <- file.path(r_results_dir, "figures")
generate_r_visualizations(cond_model, markov_model, test_probs, y_test, fig_dir)

cat("\nAll R pipeline executions completed successfully!\n")
cat("===============================================================================\n")
