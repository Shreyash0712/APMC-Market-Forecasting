"""
File: src/explainability/exogenous_gate.py
Role: Bayesian Residual Disruption Gate
Description: Dynamically modulates the ST-GNN raw forecast probabilities when verified
exogenous shocks (floods, strikes, bumper harvests) occur, preventing false crash alarms
and aligning model predictions with physical reality.
Dependencies: numpy
"""

import numpy as np

class BayesianExogenousGate:
    def __init__(self, sensitivity=1.2):
        """
        Args:
            sensitivity: Scaling factor for how aggressively exogenous shocks alter GNN logits.
        """
        self.sensitivity = sensitivity

    def update_probabilities(self, gnn_probs, supply_disruption_index, road_accessibility=1.0):
        """
        Applies a Bayesian belief update to raw GNN class probabilities.
        
        Args:
            gnn_probs: Dict {'Stable': p0, 'Crash': p1, 'Spike': p2} or array/list [p0, p1, p2].
            supply_disruption_index: Float between -1.0 (bumper glut) and +1.0 (severe destruction).
            road_accessibility: Float between 0.0 (total road blockage) and 1.0 (free transit).
            
        Returns:
            dict of updated probabilities and explanation of belief shift.
        """
        if isinstance(gnn_probs, dict):
            p_stable = gnn_probs.get("Stable", 0.33)
            p_crash = gnn_probs.get("Crash", 0.33)
            p_spike = gnn_probs.get("Spike", 0.33)
        else:
            p_stable, p_crash, p_spike = gnn_probs[0], gnn_probs[1], gnn_probs[2]

        p = np.array([p_stable, p_crash, p_spike], dtype=np.float64)
        
        # Effective disruption accounts for supply destruction and road severance
        # If roads are blocked (accessibility < 1.0), arrivals drop further (+ disruption)
        effective_shock = supply_disruption_index + (1.0 - road_accessibility) * 0.4
        effective_shock = np.clip(effective_shock, -1.0, 1.0)
        
        # Shift weights:
        # Crash is Index 1, Spike is Index 2, Stable is Index 0
        if effective_shock > 0:
            # Positive shock: Supply destruction / transit block -> Crash risk drops, Spike/Stable rises
            damping = np.exp(-self.sensitivity * effective_shock)
            p[1] = p[1] * damping
            p[0] = p[0] + (1.0 - damping) * 0.6 * p[1]
            p[2] = p[2] + (1.0 - damping) * 0.4 * p[1] + (effective_shock * 0.3)
        elif effective_shock < 0:
            # Negative shock: Bumper glut -> Crash risk spikes higher
            amplification = abs(effective_shock) * self.sensitivity
            p[1] = p[1] + (1.0 - p[1]) * min(0.95, amplification)
            p[2] = p[2] * (1.0 - abs(effective_shock))
            
        # Re-normalize to valid probability distribution
        p = np.clip(p, 1e-6, 1.0)
        p_updated = p / np.sum(p)
        
        classes = ["Stable", "Crash", "Spike"]
        new_pred = classes[int(np.argmax(p_updated))]
        original_pred = classes[int(np.argmax([p_stable, p_crash, p_spike]))]
        
        override_occurred = (new_pred != original_pred)
        
        return {
            "original_prediction": original_pred,
            "original_probabilities": {
                "Stable": round(float(p_stable), 4),
                "Crash": round(float(p_crash), 4),
                "Spike": round(float(p_spike), 4)
            },
            "updated_prediction": new_pred,
            "updated_probabilities": {
                "Stable": round(float(p_updated[0]), 4),
                "Crash": round(float(p_updated[1]), 4),
                "Spike": round(float(p_updated[2]), 4)
            },
            "override_occurred": override_occurred,
            "effective_shock": round(float(effective_shock), 4),
            "rationale": (
                f"Severe shock ({effective_shock:+.2f}) successfully overrode the GNN false alarm from {original_pred} to {new_pred}."
                if override_occurred else
                f"Shock magnitude ({effective_shock:+.2f}) was insufficient to overturn baseline forecast {original_pred}."
            )
        }
