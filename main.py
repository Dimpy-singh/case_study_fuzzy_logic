import numpy as np

# Define the triangular membership function
def triangular_mf(x, params):
    a, b, c = params
    if x <= a or x >= c:
        return 0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return (c - x) / (c - b)

# Define the T-S fuzzy model
class TSFuzzyModel:
    def __init__(self, rules):
        self.rules = rules
    
    def evaluate_rule(self, inputs, rule):
        # Evaluate membership functions for inputs
        membership_values = [triangular_mf(inputs[i], params) for i, params in enumerate(rule['antecedents'])]
        # Compute firing strength
        firing_strength = min(membership_values)
        # Compute consequent output
        consequent_output = np.dot(rule['consequent'], [1] + inputs)
        return firing_strength, consequent_output
    
    def predict(self, inputs):
        outputs = []
        for rule in self.rules:
            firing_strength, consequent_output = self.evaluate_rule(inputs, rule)
            outputs.append((firing_strength, consequent_output))
        # Compute overall output using weighted average
        weighted_outputs = [firing_strength * output for firing_strength, output in outputs]
        overall_output = sum(weighted_outputs) / sum([firing_strength for firing_strength, _ in outputs])
        return overall_output

# Define the rules of the T-S fuzzy model
rules = [
    {'antecedents': [(18, 20, 22), (-0.5, 0, 0.5)], 'consequent': [20, 1, 0.5]},  # Example rule 1
    {'antecedents': [(20, 22, 24), (-0.5, 0, 0.5)], 'consequent': [22, 1, 0.3]},  # Example rule 2
    {'antecedents': [(22, 24, 26), (-0.5, 0, 0.5)], 'consequent': [24, 1, 0.1]}   # Example rule 3
]

# Create the T-S fuzzy model
ts_model = TSFuzzyModel(rules)

# Example usage: Predict the temperature given inputs T=21, dT/dt=0.2
predicted_temperature = ts_model.predict([21, 0.2])
print("Predicted temperature:", predicted_temperature)
