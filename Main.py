import numpy as np
import matplotlib.pyplot as plt

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
        sum_firing_strengths = 0
        overall_output = 0

        # Calculate firing strengths and consequent outputs for each rule
        for rule in self.rules:
            firing_strength, consequent_output = self.evaluate_rule(inputs, rule)
            outputs.append((firing_strength, consequent_output))
            sum_firing_strengths += firing_strength

        # Calculate overall output using weighted average
        if sum_firing_strengths != 0:
            weighted_outputs = [firing_strength * output for firing_strength, output in outputs]
            overall_output = sum(weighted_outputs) / sum_firing_strengths

        return overall_output

# Define the rules of the T-S fuzzy model
rules = [
    {'antecedents': [(18, 20, 22), (-0.5, 0, 0.5)], 'consequent': [20, 1, 0.5]},  # Example rule 1
    {'antecedents': [(20, 22, 24), (-0.5, 0, 0.5)], 'consequent': [22, 1, 0.3]},  # Example rule 2
    {'antecedents': [(22, 24, 26), (-0.5, 0, 0.5)], 'consequent': [24, 1, 0.1]}   # Example rule 3
]

# Create the T-S fuzzy model
ts_model = TSFuzzyModel(rules)

# Simulation parameters
time_steps = 100
input_values_T = 20 + 2 * np.sin(np.linspace(0, 4*np.pi, time_steps))
input_values_dT_dt = np.cos(np.linspace(0, 4*np.pi, time_steps))

# Simulate and predict temperatures over time
predicted_temperatures = []
for T, dT_dt in zip(input_values_T, input_values_dT_dt):
    predicted_temperature = ts_model.predict([T, dT_dt])
    predicted_temperatures.append(predicted_temperature)

# Plot the input values and the predicted temperature
plt.figure()
plt.plot(input_values_T, label='T', color='blue')
plt.plot(input_values_dT_dt, label='dT/dt', color='orange')
plt.plot(predicted_temperatures, label='Predicted Temperature', color='green')
plt.title('Simulation of T-S Fuzzy Model Predictions with Sine Wave Inputs')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()
