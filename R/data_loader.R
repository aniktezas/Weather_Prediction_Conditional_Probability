# =========================================================================
# BAMAT207: Probability and Statistics
# Weather Prediction using Conditional Probability (R Implementation)
# Module: Data Loader & Feature Discretization
# =========================================================================

load_weather_data <- function(csv_path) {
  if (!file.exists(csv_path)) {
    stop(paste("Data file not found at:", csv_path))
  }
  
  df <- read.csv(csv_path, stringsAsFactors = FALSE)
  cat(sprintf("[R DataLoader] Loaded %d daily records from %s\n", nrow(df), csv_path))
  
  # Categorical Discretization for Probability Models
  # Humidity: Low (<60%), Medium (60-80%), High (>80%)
  df$Humidity_Cat <- ifelse(df$Humidity < 60, "Low",
                     ifelse(df$Humidity <= 80, "Medium", "High"))
  df$Humidity_Cat <- factor(df$Humidity_Cat, levels = c("Low", "Medium", "High"))
  
  # Pressure: Low (<1008 hPa), Normal (1008-1014 hPa), High (>1014 hPa)
  df$Pressure_Cat <- ifelse(df$Pressure < 1008, "Low",
                     ifelse(df$Pressure <= 1014, "Normal", "High"))
  df$Pressure_Cat <- factor(df$Pressure_Cat, levels = c("Low", "Normal", "High"))
  
  # Wind: Calm (<15 km/h), Moderate (15-28 km/h), Gusty (>28 km/h)
  df$Wind_Cat <- ifelse(df$WindSpeed < 15, "Calm",
                 ifelse(df$WindSpeed <= 28, "Moderate", "Gusty"))
  df$Wind_Cat <- factor(df$Wind_Cat, levels = c("Calm", "Moderate", "Gusty"))
  
  return(df)
}

train_test_split_temporal <- function(df, train_ratio = 0.8) {
  n <- nrow(df)
  split_idx <- floor(n * train_ratio)
  
  train_df <- df[1:split_idx, ]
  test_df <- df[(split_idx + 1):n, ]
  
  cat(sprintf("[R DataLoader] Split dataset: %d Train days (%.1f%%), %d Test days (%.1f%%)\n",
              nrow(train_df), train_ratio * 100, nrow(test_df), (1 - train_ratio) * 100))
  
  list(train = train_df, test = test_df)
}
