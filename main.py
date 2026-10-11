import os
import sys
import json
import numpy as np

# Ensure src is discoverable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import fetch_or_load_weather_data, train_test_split_temporal
from src.conditional_prob import ConditionalProbabilityModel
from src.markov_model import MarkovWeatherModel
from src.bayesian_model import GaussianBayesianClassifier
from src.comparative_models import ClimatologyBaseline, AnalyticalLogisticRegression
from src.evaluation import compute_metrics
from src.visualizer import generate_all_visualizations

def run_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    figures_dir = os.path.join(results_dir, "figures")
    
    print("=" * 75)
    print(" BAMAT207: Probability and Statistics Course Project")
    print(" Track 1: Bayesian Intelligence | Weather Prediction using Conditional Probability")
    print(" Faculty Evaluator: Dr. Pallavi Mishra")
    print("=" * 75)
    
    # 1. Data Ingestion & Preprocessing
    dataset = fetch_or_load_weather_data(data_dir)
    train_data, test_data = train_test_split_temporal(dataset, train_ratio=0.8)
    
    print(f"Dataset Size: {len(dataset['y'])} days (Train: {len(train_data['y'])}, Test: {len(test_data['y'])})")
    print(f"Overall Rain Base Rate (Prior): {np.mean(dataset['y']):.4f} ({np.sum(dataset['y'])} rainy days)")
    print("-" * 75)
    
    # 2. Model 1: Empirical Conditional Probability & Law of Total Probability
    print("[1/5] Training Empirical Conditional Probability Model...")
    cond_model = ConditionalProbabilityModel(laplace_alpha=1.0)
    cond_model.fit(train_data)
    
    tot_prob_res = cond_model.verify_law_of_total_probability(train_data)
    print(f"  -> Law of Total Probability Verification:")
    print(f"     Direct Prior P(Rain):       {tot_prob_res['empirical_prior_P(Rain)']:.4f}")
    print(f"     Reconstructed sum_h P(R|H): {tot_prob_res['reconstructed_P(Rain)']:.4f}")
    print(f"     Delta (Consistency Check):  {tot_prob_res['absolute_difference']:.6f}")
    
    # 3. Model 2: Discrete Markov Chain (1st & 2nd Order)
    print("[2/5] Training Markov Chain Weather State Transition Model...")
    markov_model = MarkovWeatherModel(laplace_alpha=1.0)
    markov_model.fit(train_data)
    T = markov_model.transition_matrix
    print(f"  -> Transition Matrix P(W_{{t+1}} | W_t):")
    print(f"     P(Dry->Dry)  = {T[0,0]:.4f} | P(Dry->Rain)  = {T[0,1]:.4f}")
    print(f"     P(Rain->Dry) = {T[1,0]:.4f} | P(Rain->Rain) = {T[1,1]:.4f}")
    print(f"  -> Markov Stationary Distribution [pi_Dry, pi_Rain]: {markov_model.stationary_distribution.round(4)}")
    
    # 4. Model 3: Gaussian Bayesian Classifier (from Scratch)
    print("[3/6] Training Gaussian Bayesian Classifier (Likelihood * Prior / Evidence)...")
    bayes_model = GaussianBayesianClassifier()
    bayes_model.fit(train_data['X'], train_data['y'])
    
    # 5. Model 4: Full Covariance Quadratic Discriminant Analysis (QDA)
    from src.bayesian_covariance_analysis import QuadraticDiscriminantAnalysis, DirichletConjugateUpdating
    print("[4/6] Training Full-Covariance QDA (Testing Naive Bayes Independence Assumption)...")
    qda_model = QuadraticDiscriminantAnalysis(reg_param=1e-3)
    qda_model.fit(train_data['X'], train_data['y'])
    
    # Dirichlet Conjugate Updating on Markov Counts
    dirichlet_model = DirichletConjugateUpdating(prior_alpha=1.0)
    expected_transitions = dirichlet_model.fit(markov_model.transition_matrix * len(train_data['y']))
    
    # 6. Model 5: Climatology Baseline & Logistic Regression
    print("[5/6] Training Benchmarks (Climatology Prior & Analytical Logistic Regression)...")
    climatology = ClimatologyBaseline()
    climatology.fit(train_data['y'])
    
    log_reg = AnalyticalLogisticRegression(lr=0.05, epochs=300)
    log_reg.fit(train_data['X'], train_data['y'])
    
    # 7. Evaluation & Inference on Test Set
    print("-" * 75)
    print("[6/6] Evaluating Models on Test Set (Future Time-Series Unseen Data)...")
    y_test = test_data['y']
    
    test_probs = {
        "Climatology Baseline": climatology.predict_proba(len(y_test)),
        "Markov Chain (1st Order)": markov_model.predict_proba(test_data),
        "Empirical Conditional Prob": cond_model.predict_proba(test_data),
        "Gaussian Naive Bayes": bayes_model.predict_proba(test_data['X']),
        "Full Covariance QDA": qda_model.predict_proba(test_data['X']),
        "Logistic Regression": log_reg.predict_proba(test_data['X'])
    }
    
    metrics = {}
    for name, probs in test_probs.items():
        preds = (probs >= 0.5).astype(int)
        metrics[name] = compute_metrics(y_test, preds, probs)
        
    # Print formatted comparative table
    print("\n" + "=" * 94)
    print(f"{'Model':<28} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'ROC-AUC':<9} | {'Brier':<9}")
    print("=" * 94)
    for name, m in metrics.items():
        print(f"{name:<28} | {m['Accuracy']:<9.4f} | {m['Precision']:<9.4f} | {m['Recall']:<9.4f} | {m['F1-Score']:<9.4f} | {m['ROC-AUC']:<9.4f} | {m['Brier Score']:<9.4f}")
    print("=" * 94)
    
    # Murphy's Brier Score Decomposition for Gaussian Bayes & QDA
    from src.brier_decomposition import murphy_brier_decomposition
    brier_decomp = murphy_brier_decomposition(y_test, test_probs["Gaussian Naive Bayes"])
    print("\n--- Murphy (1973) Brier Score Formal Decomposition (Gaussian Naive Bayes) ---")
    print(f"  Uncertainty (Climatological Difficulty): {brier_decomp['uncertainty']:.4f}")
    print(f"  Reliability (Calibration Error, lower is better): {brier_decomp['reliability']:.4f}")
    print(f"  Resolution  (Discrimination Power, higher is better): {brier_decomp['resolution']:.4f}")
    print(f"  Decomposed Brier = Rel - Res + Unc: {brier_decomp['decomposed_brier']:.4f}")
    print(f"  Brier Skill Score (BSS vs Climatology): {brier_decomp['brier_skill_score']:.4f}")
    
    # Advanced: 3-State HMM Viterbi Weather Regime Decoding
    from src.advanced_markov_hmm import GaussianHiddenMarkovModel
    hmm_model = GaussianHiddenMarkovModel(n_states=3)
    obs_test_hp = test_data['X'][:, [4, 5]] # Humidity, Pressure
    viterbi_states = hmm_model.viterbi_decode(obs_test_hp)
    regime_names = ["0:Anticyclonic", "1:Convective", "2:Frontal"]
    regime_counts = {regime_names[k]: int(np.sum(viterbi_states == k)) for k in range(3)}
    print(f"\n--- HMM Viterbi Latent Atmospheric Regimes Decoded (Test Days) ---")
    print(f"  {regime_counts}")
    
    # Advanced: Spatial Generalization Test (Pune -> Delhi)
    from src.spatial_generalization import run_spatial_cross_regional_evaluation
    spatial_res = run_spatial_cross_regional_evaluation(data_dir)
    if spatial_res:
        print(f"\n--- Cross-Regional Spatial Generalization (Trained on Pune -> Evaluated on Delhi) ---")
        print(f"  Delhi Test Sample Size: {spatial_res['delhi_test_days']} days (Base Rate: {spatial_res['delhi_prior_rain']:.4f})")
        print(f"  Empirical Conditional Probability on Delhi -> Accuracy: {spatial_res['empirical_conditional_metrics']['Accuracy']:.4f}, ROC-AUC: {spatial_res['empirical_conditional_metrics']['ROC-AUC']:.4f}")
        print(f"  Gaussian Naive Bayes on Delhi              -> Accuracy: {spatial_res['gaussian_bayes_metrics']['Accuracy']:.4f}, ROC-AUC: {spatial_res['gaussian_bayes_metrics']['ROC-AUC']:.4f}")

    # Save metrics table
    csv_lines = ["Model,Accuracy,Precision,Recall,F1-Score,ROC-AUC,Brier Score,Log-Loss\n"]
    for name, m in metrics.items():
        csv_lines.append(f"{name},{m['Accuracy']},{m['Precision']},{m['Recall']},{m['F1-Score']},{m['ROC-AUC']},{m['Brier Score']},{m['Log-Loss']}\n")
    with open(os.path.join(results_dir, "metrics_summary.csv"), "w", encoding="utf-8") as f:
        f.writelines(csv_lines)
        
    # 7. Generate Visualizations
    results_dict = {
        "cond_model": cond_model,
        "markov_model": markov_model,
        "bayes_model": bayes_model,
        "y_test": y_test,
        "test_probs": test_probs,
        "metrics": metrics
    }
    generate_all_visualizations(results_dict, figures_dir)
    
    print(f"\nAll pipeline results and plots successfully saved under: {results_dir}")
    print("=" * 75)

if __name__ == "__main__":
    run_pipeline()
