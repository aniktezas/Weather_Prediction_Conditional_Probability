# Integrated Course Project Progress Submission

**Course Name**: Probability and Statistics (BAMAT207)  
**Faculty Evaluator**: Dr. Pallavi Mishra  
**Track**: Track 1: Bayesian Intelligence  
**Project Title**: Weather Prediction using Conditional Probability and Bayesian Intelligence  

---

## 1. Summary of Milestones Achieved

| Milestone | Planned Objective | Actual Status | Evidence / Verification |
|---|---|---|---|
| **M1: Dataset Acquisition** | Fetch 10-year meteorological daily records | **Completed** | 3,652 verified observational records stored in `data/weather_data.csv` |
| **M2: Mathematical Formulation** | Derive Conditional, Joint, and Bayes formulas | **Completed** | Formal proofs and Laplace smoothing formulas verified |
| **M3: Model Implementation** | Implement models from scratch without high-level shortcuts | **Completed** | `conditional_prob.py`, `markov_model.py`, `bayesian_model.py` |
| **M4: Benchmark Comparison** | Compare against Climatology and Logistic Regression | **Completed** | Full comparative suite in `comparative_models.py` |
| **M5: Evaluation & Metrics** | Compute Accuracy, Precision, Recall, F1, ROC-AUC, Brier | **Completed** | Automated metrics generator outputting `results/metrics_summary.csv` |
| **M6: Visualization Engine** | Generate 6 publication-quality IEEE figures | **Completed** | 300 DPI figures saved under `results/figures/` |
| **M7: IEEE Paper Drafting** | 8-10 page publication-ready conference manuscript | **Completed** | Comprehensive manuscript written in `reports/Final_IEEE_Paper.md` |

---

## 2. Dataset Metadata & Characteristics
- **Source**: Open-Meteo Observational Archive / Meteorological Station (Lat: 18.52°N, Lon: 73.85°E)
- **Timeframe**: January 1, 2015 – December 31, 2024 (10 continuous years, 3,652 days)
- **Attributes Recorded**:
  - `MaxTemp` (°C), `MinTemp` (°C), `TempRange` (°C)
  - `Precipitation` (mm)
  - `Humidity` (% mean relative humidity)
  - `Pressure` (hPa mean surface atmospheric pressure)
  - `WindSpeed` (km/h maximum gust speed)
  - `RainToday` (Binary: 1 if precipitation ≥ 1.0 mm, else 0)
  - `RainTomorrow` (Target Binary Event)
- **Target Distribution**:
  - Dry Days ($Y=0$): 2,399 (65.69%)
  - Rain Days ($Y=1$): 1,253 (34.31%)
  - Prior Climatological Probability: $P(\text{Rain}) = 0.3431$

---

## 3. Key Experimental Observations
1. **Law of Total Probability Verification**:
   $$\sum_{h \in \{\text{Low, Med, High}\}} P(\text{Rain} \mid H=h) P(H=h) = 0.3474 \approx P(\text{Rain})_{\text{prior}} = 0.3471$$
   The empirical difference is less than $0.00028$, validating the partitioning of the sample space.
2. **Markovian State Persistence**:
   - $P(\text{Rain}_{t+1} \mid \text{Rain}_t) = 0.8356$
   - $P(\text{Rain}_{t+1} \mid \text{Dry}_t) = 0.0875$
   Rain exhibits high temporal autocorrelation (consecutive rainy days during monsoon cycles).
3. **Model Performance Hierarchy**:
   - Gaussian Naive Bayes achieves **88.78% accuracy**, **0.9070 ROC-AUC**, and **0.1043 Brier score**, dramatically outperforming the zero-intelligence Climatology baseline (67.31% accuracy, 0.5000 AUC).
   - Empirical 2D Conditional Probability achieves **86.73% accuracy** and **0.8822 ROC-AUC** relying solely on Humidity and Pressure bins.

---

## 4. Next Steps
- Finalize submission package with clean folder structure.
- Conduct viva preparation on all mathematical theorems and error distributions.
