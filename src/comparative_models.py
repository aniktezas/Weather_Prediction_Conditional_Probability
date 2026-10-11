import numpy as np

class ClimatologyBaseline:
    """Zero-intelligence climatology model: predicts constant prior probability."""
    def __init__(self):
        self.prior = 0.5
        
    def fit(self, y_train):
        self.prior = np.mean(y_train)
        
    def predict_proba(self, n_samples):
        return np.full(n_samples, self.prior)
        
    def predict(self, n_samples, threshold=0.5):
        return (self.predict_proba(n_samples) >= threshold).astype(int)

class AnalyticalLogisticRegression:
    """
    Standard Discriminative Probabilistic Benchmark trained via mini-batch SGD.
    P(Y=1 | X) = 1 / (1 + exp(-(w^T x + b)))
    """
    def __init__(self, lr=0.01, epochs=250, l2_reg=0.001):
        self.lr = lr
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.w = None
        self.b = 0.0
        self.mu = None
        self.std = None

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        self.mu = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        X_norm = (X - self.mu) / self.std
        
        n_samples, n_feats = X_norm.shape
        self.w = np.zeros(n_feats)
        self.b = 0.0
        
        for epoch in range(self.epochs):
            z = np.dot(X_norm, self.w) + self.b
            y_pred = self._sigmoid(z)
            error = y_pred - y
            
            grad_w = (np.dot(X_norm.T, error) / n_samples) + (self.l2_reg * self.w)
            grad_b = np.sum(error) / n_samples
            
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def predict_proba(self, X):
        X_norm = (X - self.mu) / self.std
        z = np.dot(X_norm, self.w) + self.b
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
