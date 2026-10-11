import os
import numpy as np

def run_spatial_cross_regional_evaluation(data_dir):
    """
    Evaluates spatial transferability of Conditional Probability distributions:
    Train Model on Pune (Western Ghats maritime monsoon climate)
    -> Test Model on Delhi (Northern continental semi-arid climate)
    """
    from src.data_loader import load_csv_data
    from src.conditional_prob import ConditionalProbabilityModel
    from src.bayesian_model import GaussianBayesianClassifier
    from src.evaluation import compute_metrics
    
    pune_file = os.path.join(data_dir, "weather_pune_10yr.csv")
    delhi_file = os.path.join(data_dir, "weather_delhi_10yr.csv")
    
    if not (os.path.exists(pune_file) and os.path.exists(delhi_file)):
        return None
        
    pune_data = load_csv_data(pune_file)
    delhi_data = load_csv_data(delhi_file)
    
    # Train on Pune
    cond_pune = ConditionalProbabilityModel()
    cond_pune.fit(pune_data)
    
    bayes_pune = GaussianBayesianClassifier()
    bayes_pune.fit(pune_data['X'], pune_data['y'])
    
    # Evaluate on Delhi
    delhi_y = delhi_data['y']
    cond_delhi_probs = cond_pune.predict_proba(delhi_data)
    bayes_delhi_probs = bayes_pune.predict_proba(delhi_data['X'])
    
    m_cond = compute_metrics(delhi_y, (cond_delhi_probs >= 0.5).astype(int), cond_delhi_probs)
    m_bayes = compute_metrics(delhi_y, (bayes_delhi_probs >= 0.5).astype(int), bayes_delhi_probs)
    
    return {
        "source_region": "Pune (Western Maritime Monsoon)",
        "target_region": "Delhi (Continental Semi-Arid)",
        "delhi_test_days": len(delhi_y),
        "delhi_prior_rain": float(np.mean(delhi_y)),
        "empirical_conditional_metrics": m_cond,
        "gaussian_bayes_metrics": m_bayes
    }
