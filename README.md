# Weather Prediction using Conditional Probability and Bayesian Intelligence

**Course**: Probability and Statistics (BAMAT207)  
**Track**: Track 1: Bayesian Intelligence  
**Faculty Evaluator**: Dr. Pallavi Mishra  

---

## 📌 Project Overview
This project presents an empirical and theoretical investigation into predicting precipitation events using **Conditional Probability, Bayes' Theorem, Joint Probability Distributions, and Markov Decision Chains**. Rather than treating weather forecasting merely as a black-box regression task, we model the foundational probabilistic laws governing atmospheric dynamics:
1. **Marginal and Prior Probabilities**: Climatological base rate $P(\\text{Rain})$.
2. **Empirical Conditional Probability**: Multi-attribute conditioning $P(\\text{Rain}_{t+1} \\mid \\text{Humidity}_t, \\text{Pressure}_t)$.
3. **Law of Total Probability**: Decomposition over partitioned atmospheric humidity spaces.
4. **Bayes' Theorem**: Posterior updating $P(\\text{Rain} \\mid \\mathbf{X}) \\propto P(\\mathbf{X} \\mid \\text{Rain}) \\cdot P(\\text{Rain})$.
5. **Markov Chain Transitions**: First-order and second-order time-series state transitions $P(W_{t+1} \\mid W_t)$ and stationary distributions.

---

## 🚀 Folder Structure
```
Weather_Prediction_Conditional_Probability/
├── data/
│   ├── weather_data.csv          # 10-year observational meteorological dataset (3,652 days)
│   └── dataset_summary.json      # Dataset statistics and source metadata
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Preprocessing, normalization, discretization
│   ├── conditional_prob.py       # Empirical conditional & joint probability engine
│   ├── markov_model.py           # First-order & Second-order Markov chain transitions
│   ├── bayesian_model.py         # Bayes' Theorem from scratch & Gaussian likelihoods
│   ├── comparative_models.py     # Climatology Baseline & Analytical Logistic Regression
│   ├── evaluation.py             # Accuracy, Precision, Recall, F1, ROC-AUC, Brier score
│   └── visualizer.py             # Generates all IEEE publication figures
├── results/
│   ├── figures/                  # 6 High-resolution figures (Heatmaps, ROC, Calibration)
│   └── metrics_summary.csv       # Comparative performance table
├── reports/
│   ├── Project_Proposal.md       # Project Proposal document
│   ├── Progress_Submission.md    # Progress Submission milestone document
│   ├── Viva_Defense_Guide.md     # Comprehensive viva Q&A and theoretical defense
│   └── Final_IEEE_Paper.md       # Full 8-10 page IEEE conference paper manuscript
├── main.py                       # Single command to run pipeline, train, evaluate, and save plots
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

---

## 💻 Instructions for Execution

### Option A: Running the Python Pipeline
1. Open PowerShell or Terminal and navigate to the project directory:
   ```bash
   cd "d:/sem3/Innovative Project/Weather_Prediction_Conditional_Probability"
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Python pipeline:
   ```bash
   python main.py
   ```
*Output*: Computed metrics in `results/metrics_summary.csv` and 6 high-res plots in `results/figures/`.

---

### Option B: Running the R Pipeline
1. Ensure R is installed (R 4.x). Open PowerShell or Terminal and navigate to the `R` directory:
   ```bash
   cd "d:/sem3/Innovative Project/Weather_Prediction_Conditional_Probability/R"
   ```
2. Run the master R script via `Rscript`:
   ```bash
   Rscript run_all.R
   ```
   *(Or using the full path if Rscript is not in PATH: `"C:\Program Files\R\R-4.6.1\bin\Rscript.exe" run_all.R`)*
*Output*: Computed metrics in `R_results/r_metrics_summary.csv` and publication figures in `R_results/figures/`.

