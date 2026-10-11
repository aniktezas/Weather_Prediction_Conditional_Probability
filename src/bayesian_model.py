import numpy as np

class GaussianBayesianClassifier:
    """
    Full Bayesian Model implemented from mathematical first principles.
    Computes Prior, Gaussian Likelihood, Evidence, and Posterior Distributions.
    
    Bayes' Rule:
    P(Y = c | X) = [ P(Y = c) * prod_j N(x_j; mu_cj, sigma_cj^2) ] / P(X)
    """
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
        self.classes = np.array([0, 1])
        self.priors = {}
        self.means = {}
        self.variances = {}

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        for c in self.classes:
            X_c = X_train[y_train == c]
            self.priors[c] = len(X_c) / n_samples
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0) + self.epsilon

    def _gaussian_log_likelihood(self, x, mean, var):
        # log N(x; mu, sigma^2) = -0.5 * log(2*pi*sigma^2) - ((x - mu)^2) / (2*sigma^2)
        exponent = -((x - mean) ** 2) / (2.0 * var)
        normalization = -0.5 * np.log(2.0 * np.pi * var)
        return normalization + exponent

    def predict_proba(self, X):
        """
        Computes exact posterior probabilities P(Y=1 | X) and P(Y=0 | X)
        using log-sum-exp stabilization.
        """
        n_samples = X.shape[0]
        posteriors = np.zeros((n_samples, 2))
        
        for i in range(n_samples):
            x = X[i]
            log_post = np.zeros(2)
            for c in [0, 1]:
                log_prior = np.log(self.priors[c])
                log_lik = np.sum(self._gaussian_log_likelihood(x, self.means[c], self.variances[c]))
                log_post[c] = log_prior + log_lik
                
            # Log-sum-exp normalization
            max_log = np.max(log_post)
            exp_terms = np.exp(log_post - max_log)
            probs = exp_terms / np.sum(exp_terms)
            posteriors[i] = probs
            
        return posteriors[:, 1] # Return probability of Rain (c=1)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
