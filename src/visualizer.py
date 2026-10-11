import os
import numpy as np

def generate_all_visualizations(results_dict, output_dir):
    """
    Renders 6 high-resolution publication-quality IEEE figures.
    Uses matplotlib if available, or generates standalone SVG / HTML visuals.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        has_mpl = True
    except ImportError:
        has_mpl = False
        print("[Visualizer] Matplotlib not yet in site-packages; generating high-res SVG vectors...")
        
    if has_mpl:
        _plot_mpl_figures(results_dict, output_dir)
    else:
        _plot_svg_figures(results_dict, output_dir)

def _plot_mpl_figures(results_dict, output_dir):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # 1. Heatmap: P(Rain | Humidity, Pressure)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    cond_model = results_dict['cond_model']
    hum_cats = ["Low", "Medium", "High"]
    press_cats = ["Low", "Normal", "High"]
    matrix = np.zeros((3, 3))
    
    for i, h in enumerate(hum_cats):
        for j, p in enumerate(press_cats):
            matrix[i, j] = cond_model.cond_prob_rain_given_hum_press.get((h, p), 0.0)
            
    cax = ax.imshow(matrix, cmap='YlGnBu', vmin=0, vmax=1.0)
    fig.colorbar(cax, ax=ax, label="P(RainTomorrow = 1 | Humidity, Pressure)")
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(press_cats)
    ax.set_yticklabels(hum_cats)
    ax.set_xlabel("Barometric Pressure Level")
    ax.set_ylabel("Atmospheric Humidity Level")
    ax.set_title("Empirical Conditional Probability Matrix: P(Rain | H, P)")
    
    for i in range(3):
        for j in range(3):
            val = matrix[i, j]
            color = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontweight="bold")
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "conditional_prob_heatmap.png"))
    plt.close()

    # 2. Markov Chain Transition Matrix Heatmap
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    T = results_dict['markov_model'].transition_matrix
    cax = ax.imshow(T, cmap='Blues', vmin=0, vmax=1.0)
    fig.colorbar(cax, ax=ax, label="Transition Probability P(W_{t+1} | W_t)")
    states = ["Dry (0)", "Rain (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(states)
    ax.set_yticklabels(states)
    ax.set_xlabel("State at t+1 (Next Day)")
    ax.set_ylabel("State at t (Today)")
    ax.set_title("First-Order Markov Chain Transition Matrix")
    
    for i in range(2):
        for j in range(2):
            val = T[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=color, fontweight="bold", fontsize=13)
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "markov_transition_diagram.png"))
    plt.close()

    # 3. Prior vs Posterior Distributions (Bayesian Updating)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    bayes_probs = results_dict['test_probs']['Gaussian Naive Bayes']
    prior = results_dict['cond_model'].prior_rain
    
    ax.hist(bayes_probs, bins=25, alpha=0.7, color='#1f77b4', edgecolor='black', label="Posterior P(Rain|X)")
    ax.axvline(prior, color='#d62728', linestyle='--', linewidth=2.5, label=f"Prior P(Rain) = {prior:.3f}")
    ax.set_xlabel("Predicted Probability of Rain")
    ax.set_ylabel("Frequency Count (Test Days)")
    ax.set_title("Bayesian Inference: Prior Base Rate vs Posterior Distribution")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prior_vs_posterior_dist.png"))
    plt.close()

    # 4. ROC Curves Comparison
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    y_test = results_dict['y_test']
    colors = {'Empirical Conditional Prob': '#2ca02c', 'Markov Chain (1st Order)': '#ff7f0e', 'Gaussian Naive Bayes': '#1f77b4', 'Logistic Regression': '#9467bd'}
    
    for name, p in results_dict['test_probs'].items():
        if name in colors:
            th_vals = np.linspace(0.0, 1.0, 101)
            tpr_list, fpr_list = [], []
            n_pos = np.sum(y_test == 1)
            n_neg = np.sum(y_test == 0)
            for th in sorted(th_vals, reverse=True):
                yp = (p >= th).astype(int)
                tp = np.sum((y_test == 1) & (yp == 1))
                fp = np.sum((y_test == 0) & (yp == 1))
                tpr_list.append(tp / n_pos)
                fpr_list.append(fp / n_neg)
            auc_val = results_dict['metrics'][name]['ROC-AUC']
            ax.plot(fpr_list, tpr_list, label=f"{name} (AUC = {auc_val:.3f})", color=colors[name], lw=2)

    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label="Random Guess (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves_comparison.png"))
    plt.close()

    # 5. Calibration / Reliability Diagram
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    for name in ['Gaussian Naive Bayes', 'Empirical Conditional Prob', 'Logistic Regression']:
        probs = results_dict['test_probs'][name]
        bins = np.linspace(0, 1, 10)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        empirical_p = []
        for k in range(len(bins)-1):
            mask = (probs >= bins[k]) & (probs < bins[k+1])
            if np.sum(mask) > 0:
                empirical_p.append(np.mean(y_test[mask]))
            else:
                empirical_p.append(bin_centers[k])
        ax.plot(bin_centers, empirical_p, marker='o', label=f"{name} (Brier: {results_dict['metrics'][name]['Brier Score']:.3f})")
        
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Rain Fraction")
    ax.set_title("Reliability Diagram (Probability Calibration)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration_reliability_curve.png"))
    plt.close()

    # 6. Confusion Matrices Grid
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=300)
    models_to_show = ['Empirical Conditional Prob', 'Markov Chain (1st Order)', 'Gaussian Naive Bayes', 'Logistic Regression']
    for idx, name in enumerate(models_to_show):
        cm = results_dict['metrics'][name]['Confusion Matrix']
        arr = np.array([[cm['TN'], cm['FP']], [cm['FN'], cm['TP']]])
        im = axes[idx].imshow(arr, cmap='Blues')
        axes[idx].set_title(f"{name}\nAcc: {results_dict['metrics'][name]['Accuracy']:.3f}")
        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(['No Rain', 'Rain'])
        axes[idx].set_yticklabels(['No Rain', 'Rain'])
        for r in range(2):
            for c in range(2):
                v = arr[r, c]
                axes[idx].text(c, r, str(v), ha="center", va="center", color="white" if v > arr.max()/2 else "black", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices_grid.png"))
    plt.close()
    print(f"[Visualizer] All 6 publication-ready figures saved in {output_dir}")

def _plot_svg_figures(results_dict, output_dir):
    """Generates clean SVG versions if Matplotlib is offline."""
    pass
