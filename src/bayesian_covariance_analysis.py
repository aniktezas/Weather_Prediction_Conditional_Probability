import numpy as np

class QuadraticDiscriminantAnalysis:
    """
    Full-Covariance Gaussian Generative Classifier.
    Eliminates Naive Bayes conditional independence assumption by modeling
    the complete covariance matrix Sigma_c for each class:
    P(x | Y=c) ~ N(mu_c, Sigma_c)
    """
    def __init__(self, reg_param=1e-4):
        self.reg_param = reg_param
        self.priors = {}
        self.means = {}
        self.covariances = {}
        self.inv_covariances = {}
        self.log_det_covariances = {}

    def fit(self, X, y):
        n_samples, n_features = X.shape
        classes = np.unique(y)
        for c in classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / n_samples
            mu_c = np.mean(X_c, axis=0)
            self.means[c] = mu_c
            
            # Full covariance with ridge regularization
            cov_c = np.cov(X_c, rowvar=False) + (self.reg_param * np.eye(n_features))
            self.covariances[c] = cov_c
            
            # Inversion and log-determinant via Cholesky or SVD
            sign, logdet = np.linalg.slogdet(cov_c)
            self.log_det_covariances[c] = logdet
            self.inv_covariances[c] = np.linalg.pinv(cov_c)

    def predict_proba(self, X):
        n_samples = X.shape[0]
        probs = np.zeros(n_samples)
        
        for i in range(n_samples):
            x = X[i]
            log_post = np.zeros(2)
            for c in [0, 1]:
                diff = x - self.means[c]
                # Mahalanobis quadratic form: diff^T * Sigma_c^{-1} * diff
                maha = np.dot(np.dot(diff, self.inv_covariances[c]), diff)
                log_lik = -0.5 * (self.log_det_covariances[c] + maha + len(x) * np.log(2.0 * np.pi))
                log_post[c] = np.log(self.priors[c]) + log_lik
                
            max_log = np.max(log_post)
            exp_terms = np.exp(log_post - max_log)
            p = exp_terms / np.sum(exp_terms)
            probs[i] = p[1]
            
        return probs

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

class DirichletConjugateUpdating:
    """
    Bayesian Conjugate Updating for Categorical Transition Probabilities.
    Prior: Dir(alpha_0)
    Likelihood: Multinomial(n)
    Posterior: Dir(alpha_0 + n)
    Expected Transition Probability: E[P_ij] = (n_ij + alpha_ij) / sum_k (n_ik + alpha_ik)
    """
    def __init__(self, prior_alpha=1.0):
        self.prior_alpha = prior_alpha
        self.posterior_params = None
        self.expected_transitions = None

    def fit(self, transitions):
        # transitions is 2x2 count matrix
        self.posterior_params = transitions + self.prior_alpha
        row_sums = self.posterior_params.sum(axis=1, keepdims=True)
        self.expected_transitions = self.posterior_params / row_sums
        return self.expected_transitions
