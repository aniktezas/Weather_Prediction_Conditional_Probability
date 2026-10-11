# =========================================================================
# BAMAT207: Probability and Statistics
# Module: Publication-Quality R Visualizations (300 DPI PNGs)
# =========================================================================

generate_r_visualizations <- function(cond_model, markov_model, test_probs, y_test, out_dir) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  # 1. Heatmap: P(Rain | Humidity, Pressure)
  png(file.path(out_dir, "r_conditional_prob_heatmap.png"), width = 2100, height = 1600, res = 300)
  par(mar = c(5, 5, 4, 6))
  M <- cond_model$cond_matrix
  pal <- colorRampPalette(c("#edf8b1", "#7fcdbb", "#1d91c0", "#081d58"))(100)
  image(1:3, 1:3, t(M), col = pal, axes = FALSE, xlab = "Barometric Pressure Level", ylab = "Atmospheric Humidity Level",
        main = "R: Empirical Conditional Probability Matrix P(Rain | H, P)")
  axis(1, at = 1:3, labels = colnames(M))
  axis(2, at = 1:3, labels = rownames(M))
  box()
  for (i in 1:3) {
    for (j in 1:3) {
      val <- M[i, j]
      col_txt <- ifelse(val > 0.55, "white", "black")
      text(j, i, sprintf("%.3f", val), col = col_txt, font = 2, cex = 1.2)
    }
  }
  dev.off()
  
  # 2. Markov Chain Transition Matrix
  png(file.path(out_dir, "r_markov_transition_diagram.png"), width = 1800, height = 1500, res = 300)
  par(mar = c(5, 5, 4, 5))
  T_mat <- markov_model$transition_matrix
  pal_blue <- colorRampPalette(c("#f7fbff", "#6baed6", "#08519c"))(100)
  image(1:2, 1:2, t(T_mat), col = pal_blue, axes = FALSE,
        xlab = "State at t+1 (Next Day)", ylab = "State at t (Today)",
        main = "R: Discrete Markov Transition Matrix P(W_{t+1} | W_t)")
  axis(1, at = 1:2, labels = c("Dry", "Rain"))
  axis(2, at = 1:2, labels = c("Dry", "Rain"))
  box()
  for (i in 1:2) {
    for (j in 1:2) {
      v <- T_mat[i, j]
      col_txt <- ifelse(v > 0.5, "white", "black")
      text(j, i, sprintf("%.4f", v), col = col_txt, font = 2, cex = 1.4)
    }
  }
  dev.off()
  
  # 3. Prior vs Posterior Distribution Shift
  png(file.path(out_dir, "r_prior_vs_posterior_dist.png"), width = 2100, height = 1500, res = 300)
  par(mar = c(5, 5, 4, 2))
  bayes_p <- test_probs$`Gaussian Naive Bayes`
  hist(bayes_p, breaks = 25, col = rgb(0.2, 0.4, 0.8, 0.6), border = "white",
       xlab = "Predicted Posterior Probability of Rain P(Rain | X)",
       ylab = "Frequency Count (Unseen Test Days)",
       main = "R: Bayesian Inference - Prior Base Rate vs Posterior Distribution")
  abline(v = cond_model$prior_rain, col = "red", lwd = 3, lty = 2)
  legend("topright", legend = c(sprintf("Prior Base Rate P(Rain) = %.3f", cond_model$prior_rain), "Posterior P(Rain | X)"),
         col = c("red", rgb(0.2, 0.4, 0.8, 0.6)), lty = c(2, 1), lwd = c(3, 8), bty = "n")
  dev.off()
  
  # 4. ROC Curves Comparison
  png(file.path(out_dir, "r_roc_curves_comparison.png"), width = 2100, height = 1600, res = 300)
  par(mar = c(5, 5, 4, 2))
  plot(c(0, 1), c(0, 1), type = "l", lty = 2, col = "gray40", lwd = 1.5,
       xlab = "False Positive Rate (FPR)", ylab = "True Positive Rate (TPR)",
       main = "R: ROC Curve Comparison Across Probabilistic Models")
       
  cols <- c("Climatology Baseline" = "gray60", "Markov Chain (1st Order)" = "#e66101",
            "Empirical Conditional Prob" = "#2b83ba", "Gaussian Naive Bayes" = "#7b3294",
            "Logistic Regression" = "#008837")
            
  th_seq <- seq(1.0, 0.0, length.out = 101)
  n_pos <- sum(y_test == 1)
  n_neg <- sum(y_test == 0)
  
  for (m_name in names(test_probs)) {
    p <- test_probs[[m_name]]
    tpr_v <- sapply(th_seq, function(th) sum(y_test == 1 & p >= th) / n_pos)
    fpr_v <- sapply(th_seq, function(th) sum(y_test == 0 & p >= th) / n_neg)
    lines(fpr_v, tpr_v, col = cols[m_name], lwd = 2.5)
  }
  legend("bottomright", legend = names(cols), col = cols, lwd = 2.5, bty = "n")
  dev.off()
  
  cat(sprintf("[R Visualizer] All publication figures successfully generated in: %s\n", out_dir))
}
