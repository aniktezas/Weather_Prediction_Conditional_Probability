import numpy as np

class ConditionalProbabilityModel:
    """
    Mathematical Implementation of Empirical Conditional Probability,
    Joint Probability Distributions, and the Law of Total Probability.
    """
    def __init__(self, laplace_alpha=1.0):
        self.alpha = laplace_alpha
        self.prior_rain = None
        self.prior_dry = None
        self.prob_rain_given_hum = {}
        self.prob_rain_given_press = {}
        self.prob_rain_given_wind = {}
        self.joint_prob_hum_press_rain = {}
        self.joint_prob_hum_press = {}
        self.cond_prob_rain_given_hum_press = {}

    def fit(self, train_data):
        y = train_data['y']
        n_total = len(y)
        n_rain = np.sum(y == 1)
        n_dry = np.sum(y == 0)
        
        # 1. Prior Probabilities
        self.prior_rain = n_rain / n_total
        self.prior_dry = n_dry / n_total
        
        hum = train_data['hum_cat']
        press = train_data['press_cat']
        wind = train_data['wind_cat']
        
        # 2. 1D Conditional Probabilities P(Rain=1 | Feature=val)
        for cat in ["Low", "Medium", "High"]:
            mask = (hum == cat)
            cnt_cat = np.sum(mask)
            cnt_rain_cat = np.sum(y[mask] == 1)
            # Laplace smoothing: (count + alpha) / (total + 2*alpha)
            self.prob_rain_given_hum[cat] = (cnt_rain_cat + self.alpha) / (cnt_cat + 2 * self.alpha)
            
        for cat in ["Low", "Normal", "High"]:
            mask = (press == cat)
            cnt_cat = np.sum(mask)
            cnt_rain_cat = np.sum(y[mask] == 1)
            self.prob_rain_given_press[cat] = (cnt_rain_cat + self.alpha) / (cnt_cat + 2 * self.alpha)

        for cat in ["Calm", "Moderate", "Gusty"]:
            mask = (wind == cat)
            cnt_cat = np.sum(mask)
            cnt_rain_cat = np.sum(y[mask] == 1)
            self.prob_rain_given_wind[cat] = (cnt_rain_cat + self.alpha) / (cnt_cat + 2 * self.alpha)

        # 3. 2D Joint & Multi-Condition: P(Rain=1 | Humidity=h, Pressure=p)
        for h in ["Low", "Medium", "High"]:
            for p in ["Low", "Normal", "High"]:
                pair_mask = (hum == h) & (press == p)
                n_pair = np.sum(pair_mask)
                n_rain_pair = np.sum(y[pair_mask] == 1)
                
                # Empirical Joint Probability P(H=h, P=p, R=1)
                self.joint_prob_hum_press_rain[(h, p)] = n_rain_pair / n_total
                # Marginal Joint P(H=h, P=p)
                self.joint_prob_hum_press[(h, p)] = n_pair / n_total
                
                # Multi-condition: P(Rain=1 | H=h, P=p)
                p_cond = (n_rain_pair + self.alpha) / (n_pair + 2 * self.alpha)
                self.cond_prob_rain_given_hum_press[(h, p)] = p_cond

    def verify_law_of_total_probability(self, train_data):
        """
        Verifies Law of Total Probability:
        P(Rain) = sum_h P(Rain | Humidity=h) * P(Humidity=h)
        """
        hum = train_data['hum_cat']
        n_total = len(hum)
        reconstructed_p_rain = 0.0
        details = []
        
        for cat in ["Low", "Medium", "High"]:
            p_h = np.sum(hum == cat) / n_total
            p_r_given_h = self.prob_rain_given_hum[cat]
            contrib = p_r_given_h * p_h
            reconstructed_p_rain += contrib
            details.append({"category": cat, "P(H)": p_h, "P(Rain|H)": p_r_given_h, "Contribution": contrib})
            
        return {
            "empirical_prior_P(Rain)": self.prior_rain,
            "reconstructed_P(Rain)": reconstructed_p_rain,
            "absolute_difference": abs(self.prior_rain - reconstructed_p_rain),
            "step_contributions": details
        }

    def predict_proba(self, test_data):
        """Returns P(RainTomorrow=1 | Humidity, Pressure)."""
        hum = test_data['hum_cat']
        press = test_data['press_cat']
        probs = []
        for h, p in zip(hum, press):
            prob = self.cond_prob_rain_given_hum_press.get((h, p), self.prior_rain)
            probs.append(prob)
        return np.array(probs)

    def predict(self, test_data, threshold=0.5):
        probs = self.predict_proba(test_data)
        return (probs >= threshold).astype(int)
