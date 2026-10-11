import numpy as np

class MarkovWeatherModel:
    """
    First-Order and Second-Order Discrete-Time Markov Chain Models.
    Weather States: 0 = Dry (No Rain), 1 = Rain.
    """
    def __init__(self, laplace_alpha=1.0):
        self.alpha = laplace_alpha
        self.transition_matrix = None # 2x2 matrix: P(W_{t+1} = j | W_t = i)
        self.second_order_matrix = None # 4x2 matrix: P(W_{t+1} | W_t, W_{t-1})
        self.stationary_distribution = None

    def fit(self, train_data):
        y = train_data['y'] # Target RainTomorrow
        rain_today = train_data['RainToday'] # State W_t
        
        # 1. First-Order Transition Matrix: P[i, j] = P(W_{t+1}=j | W_t=i)
        counts = np.zeros((2, 2))
        for i in range(len(rain_today)):
            s_t = rain_today[i]
            s_next = y[i]
            counts[s_t, s_next] += 1
            
        # Add Laplace smoothing
        counts_smoothed = counts + self.alpha
        row_sums = counts_smoothed.sum(axis=1, keepdims=True)
        self.transition_matrix = counts_smoothed / row_sums
        
        # 2. Compute Stationary Distribution pi = pi * P
        # Solved via eigenvector corresponding to eigenvalue 1
        eigenvals, eigenvecs = np.linalg.eig(self.transition_matrix.T)
        idx = np.argmin(np.abs(eigenvals - 1.0))
        stationary = np.real(eigenvecs[:, idx])
        self.stationary_distribution = stationary / np.sum(stationary)
        
        # 3. Second-Order Markov Chain: P(W_{t+1} | W_t, W_{t-1})
        sec_counts = np.zeros((4, 2)) # Rows: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        for i in range(1, len(rain_today)):
            prev = rain_today[i - 1]
            curr = rain_today[i]
            nxt = y[i]
            row_idx = (prev * 2) + curr
            sec_counts[row_idx, nxt] += 1
            
        sec_smoothed = sec_counts + self.alpha
        self.second_order_matrix = sec_smoothed / sec_smoothed.sum(axis=1, keepdims=True)

    def multi_step_forecast(self, current_state, k_days=7):
        """Computes k-step transition probabilities P^k."""
        Pk = np.linalg.matrix_power(self.transition_matrix, k_days)
        # Probabilities of [Dry, Rain] given current_state
        return Pk[current_state]

    def predict_proba(self, test_data):
        """Predicts P(RainTomorrow = 1 | RainToday)."""
        rain_today = test_data['RainToday']
        probs = [self.transition_matrix[s, 1] for s in rain_today]
        return np.array(probs)

    def predict(self, test_data, threshold=0.5):
        probs = self.predict_proba(test_data)
        return (probs >= threshold).astype(int)
