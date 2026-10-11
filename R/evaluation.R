# =========================================================================
# BAMAT207: Probability and Statistics
# Module: Model Evaluation & Scoring Rules (Brier Score, ROC-AUC, Log-Loss)
# =========================================================================

compute_metrics_r <- function(y_true, y_probs, threshold = 0.5) {
  y_pred <- ifelse(y_probs >= threshold, 1, 0)
  
  tp <- sum(y_true == 1 & y_pred == 1)
  fp <- sum(y_true == 0 & y_pred == 1)
  fn <- sum(y_true == 1 & y_pred == 0)
  tn <- sum(y_true == 0 & y_pred == 0)
  total <- length(y_true)
  
  acc <- (tp + tn) / total
  prec <- ifelse((tp + fp) > 0, tp / (tp + fp), 0.0)
  rec <- ifelse((tp + fn) > 0, tp / (tp + fn), 0.0)
  f1 <- ifelse((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
  
  # Brier Score (Mean Squared Error in probability space)
  brier_score <- mean((y_probs - y_true)^2)
  
  # Log Loss
  eps <- 1e-12
  p_clip <- pmax(pmin(y_probs, 1 - eps), eps)
  log_loss <- -mean(y_true * log(p_clip) + (1 - y_true) * log(1 - p_clip))
  
  # Fast Trapezoidal ROC-AUC
  th_seq <- seq(1.0, 0.0, length.out = 101)
  tpr_vec <- numeric(length(th_seq))
  fpr_vec <- numeric(length(th_seq))
  n_pos <- sum(y_true == 1)
  n_neg <- sum(y_true == 0)
  
  for (k in 1:length(th_seq)) {
    yp <- ifelse(y_probs >= th_seq[k], 1, 0)
    tpr_vec[k] <- sum(y_true == 1 & yp == 1) / n_pos
    fpr_vec[k] <- sum(y_true == 0 & yp == 1) / n_neg
  }
  
  # Trapezoid integration
  roc_auc <- 0.0
  for (k in 1:(length(fpr_vec) - 1)) {
    dx <- fpr_vec[k + 1] - fpr_vec[k]
    my <- (tpr_vec[k + 1] + tpr_vec[k]) / 2
    roc_auc <- roc_auc + (dx * my)
  }
  roc_auc <- max(0.0, min(1.0, roc_auc))
  
  list(
    Accuracy = round(acc, 4),
    Precision = round(prec, 4),
    Recall = round(rec, 4),
    F1 = round(f1, 4),
    ROC_AUC = round(roc_auc, 4),
    Brier_Score = round(brier_score, 4),
    Log_Loss = round(log_loss, 4),
    TP = tp, FP = fp, FN = fn, TN = tn
  )
}
