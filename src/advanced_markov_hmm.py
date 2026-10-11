import numpy as np

class GaussianHiddenMarkovModel:
    """
    3-State Hidden Markov Model (HMM) for Synoptic Atmospheric Regimes:
    - State 0: Stable Anticyclonic (Clear, subsiding air)
    - State 1: Moist Convective (Humid, transitional)
    - State 2: Frontal Saturated (Active precipitation)
    
    Emission Densities: P(x_t | S_t = k) ~ N(mu_k, sigma_k^2)
    Inference: Viterbi Algorithm for most probable latent weather regime path.
    """
    def __init__(self, n_states=3):
        self.n_states = n_states
        # Initial state probabilities
        self.pi = np.array([0.50, 0.30, 0.20])
        # Latent transition matrix A[i, j] = P(S_{t+1}=j | S_t=i)
        self.A = np.array([
            [0.85, 0.12, 0.03],
            [0.20, 0.65, 0.15],
            [0.08, 0.22, 0.70]
        ])
        # Emission parameters on Humidity & Pressure
        self.means = np.array([
            [48.0, 1013.0], # State 0: Dry, High pressure
            [72.0, 1008.0], # State 1: Moderate humidity, Normal pressure
            [88.0, 1003.0]  # State 2: High humidity, Low cyclonic pressure
        ])
        self.vars = np.array([
            [80.0, 12.0],
            [60.0, 10.0],
            [45.0, 15.0]
        ])

    def _emission_log_prob(self, x, state_k):
        mu = self.means[state_k]
        var = self.vars[state_k]
        exponent = -0.5 * np.sum(((x - mu) ** 2) / var)
        norm = -0.5 * np.sum(np.log(2.0 * np.pi * var))
        return norm + exponent

    def viterbi_decode(self, obs_seq):
        """
        Viterbi Algorithm: Computes the most likely sequence of hidden atmospheric regimes.
        obs_seq: array of shape (T, 2) containing [Humidity, Pressure]
        """
        T = len(obs_seq)
        V = np.zeros((T, self.n_states))
        backpointer = np.zeros((T, self.n_states), dtype=int)
        
        # Initialization
        for s in range(self.n_states):
            V[0, s] = np.log(self.pi[s]) + self._emission_log_prob(obs_seq[0], s)
            
        # Induction
        for t in range(1, T):
            for s in range(self.n_states):
                emit = self._emission_log_prob(obs_seq[t], s)
                trans_probs = V[t - 1] + np.log(self.A[:, s])
                best_prev = np.argmax(trans_probs)
                V[t, s] = trans_probs[best_prev] + emit
                backpointer[t, s] = best_prev
                
        # Backtracking
        best_path = np.zeros(T, dtype=int)
        best_path[-1] = np.argmax(V[-1])
        for t in range(T - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]
            
        return best_path
