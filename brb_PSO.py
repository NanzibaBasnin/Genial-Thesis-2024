import os
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from pyswarm import pso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Constants for Alzheimer's diagnosis
AD_SCORE = 1.0
MCI_SCORE = 0.7
CN_SCORE = 0.2

# Load the CSV file
file_path = 'C:\\Users\\HP\\Documents\\Discertation\\Code\\BRB_PSO\\selected_columns.csv'

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file at {file_path} does not exist. Please check the path and try again.")

df = pd.read_csv(file_path)

# Select only the necessary columns
df = df[['Crisp_Value', 'Age', 'Group']]

# Split the data into training and test sets
train_size = 0.8
test_size = 0.2

train, test = train_test_split(df, train_size=train_size, shuffle=True, random_state=42)

# Split training data into three clients
client_data = np.array_split(train, 3)

# Scaling the features
scaler = MinMaxScaler()
for i in range(3):
    client_data[i][['Crisp_Value', 'Age']] = scaler.fit_transform(client_data[i][['Crisp_Value', 'Age']])

test[['Crisp_Value', 'Age']] = scaler.transform(test[['Crisp_Value', 'Age']])

# Save split data into CSV files
output_dir = 'C:\\Users\\HP\\Documents\\Discertation\\Code\\BRB_PSO\\'
os.makedirs(output_dir, exist_ok=True)

for i in range(3):
    client_data[i].to_csv(os.path.join(output_dir, f'client_{i+1}_train_set.csv'), index=False)
test.to_csv(os.path.join(output_dir, 'test_set_POS.csv'), index=False)

# Define the BRB model
def transform_input(value, ref_high, ref_low):
    if value >= ref_high:
        return 1, 0
    elif value <= ref_low:
        return 0, 1
    else:
        high_belief = (ref_high - value) / (ref_high - ref_low)
        low_belief = 1 - high_belief
        return high_belief, low_belief

def calculate_matching_degree(crisp_value, age, belief_degrees, rule_idx):
    matching_degree = 1.0
    if rule_idx % 3 == 0:
        high_belief, low_belief = transform_input(crisp_value, 1, 0)
    elif rule_idx % 3 == 1:
        high_belief, low_belief = transform_input(age, 1, 0)
    else:
        high_belief, low_belief = transform_input((crisp_value + age) / 2, 1, 0)
    
    for i in range(3):
        matching_degree *= (belief_degrees[rule_idx][i] ** high_belief) * ((1 - belief_degrees[rule_idx][i]) ** low_belief)
    
    return matching_degree

def update_beliefs(matching_degrees, belief_degrees, rule_weights):
    activation_weights = matching_degrees * rule_weights
    sum_activation_weights = np.sum(activation_weights)
    if sum_activation_weights == 0:
        activation_weights = np.ones_like(activation_weights) / len(activation_weights)
    else:
        activation_weights /= sum_activation_weights
    updated_beliefs = np.sum(activation_weights[:, np.newaxis] * belief_degrees, axis=0)
    return updated_beliefs

def aggregate_output(beliefs):
    return np.dot(beliefs, [CN_SCORE, MCI_SCORE, AD_SCORE])

def brb_model(parameters, inputs):
    belief_degrees = parameters[:27].reshape((9, 3))
    rule_weights = parameters[29:38]
    attribute_weights = parameters[27:29]
    combined_beliefs = np.zeros(3)
    matching_degrees = np.zeros(9)

    crisp_value, age = inputs
    for rule_idx in range(9):
        matching_degree_crisp = calculate_matching_degree(crisp_value, age, belief_degrees, rule_idx)
        matching_degree_age = calculate_matching_degree(age, crisp_value, belief_degrees, rule_idx)
        matching_degrees[rule_idx] = (matching_degree_crisp ** attribute_weights[0]) * \
                                     (matching_degree_age ** attribute_weights[1])

    combined_beliefs = update_beliefs(matching_degrees, belief_degrees, rule_weights)
    return combined_beliefs

# Ensure belief degrees sum to 1
def ensure_belief_degrees(parameters):
    for i in range(9):
        sum_belief = sum(parameters[i*3:(i+1)*3])
        if sum_belief == 0:
            parameters[i*3:(i+1)*3] = [1/3, 1/3, 1/3]  # Assign equal belief if sum is zero
        else:
            for j in range(3):
                parameters[i*3+j] /= sum_belief
    return parameters

# Initialize storage for rule weights and belief degrees
rule_weights_beliefs_history = []

# Objective function for optimization
def objective_function(parameters, client_train_data):
    parameters = ensure_belief_degrees(parameters)
    total_error = 0
    for _, row in client_train_data.iterrows():
        inputs = row[['Crisp_Value', 'Age']].values
        observed_output = row['Group']
        predicted_beliefs = brb_model(parameters, inputs)
        predicted_output = aggregate_output(predicted_beliefs)
        total_error += (predicted_output - observed_output) ** 2
    return total_error

# Define the bounds for the parameters
bounds = [(0, 1)] * 38
lb = [0] * 38
ub = [1] * 38

# Run PSO for each client
optimized_parameters_clients = []
for i in range(3):
    print(f"Starting PSO for Client {i+1}...")
    start_time = time.time()
    optimized_parameters_pso, _ = pso(lambda x: objective_function(x, client_data[i]), lb, ub, swarmsize=50, maxiter=100)
    optimized_parameters_clients.append(optimized_parameters_pso)
    print(f"PSO for Client {i+1} completed in {time.time() - start_time} seconds.")

# Average the optimized parameters
averaged_parameters = np.mean(optimized_parameters_clients, axis=0)

# Run PSO again on the averaged parameters to get the final optimized parameters
print("Starting final PSO on averaged parameters...")
start_time = time.time()
final_optimized_parameters_pso, _ = pso(lambda x: objective_function(x, train), lb, ub, swarmsize=50, maxiter=100)
print(f"Final PSO completed in {time.time() - start_time} seconds.")

# Evaluate the optimized model on the test set
def evaluate_model(parameters, data):
    total_error = 0
    predicted_groups = []
    for _, row in data.iterrows():
        inputs = row[['Crisp_Value', 'Age']].values
        observed_output = row['Group']
        predicted_beliefs = brb_model(parameters, inputs)
        predicted_output = aggregate_output(predicted_beliefs)
        predicted_groups.append(predicted_output)
        total_error += (predicted_output - observed_output) ** 2
    mse = total_error / len(data)
    return mse, predicted_groups

# Calculate accuracy
def calculate_accuracy(original, predicted, threshold=0.1):
    correct_predictions = np.abs(original - predicted) <= threshold
    return np.mean(correct_predictions)

# Evaluate the model on the training and test sets
train_mse, train_predicted = evaluate_model(final_optimized_parameters_pso, train)
test_mse, test_predicted = evaluate_model(final_optimized_parameters_pso, test)

train_accuracy = calculate_accuracy(train['Group'], train_predicted)
test_accuracy = calculate_accuracy(test['Group'], test_predicted)

print(f"Training MSE: {train_mse}")
print(f"Test MSE: {test_mse}")
print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Save the predicted and observed values to CSV files
train_results = pd.DataFrame({'Observed': train['Group'], 'Predicted': train_predicted})
test_results = pd.DataFrame({'Observed': test['Group'], 'Predicted': test_predicted})

train_results.to_csv(os.path.join(output_dir, 'train_results_POS1.csv'), index=False)
test_results.to_csv(os.path.join(output_dir, 'test_results_POS1.csv'), index=False)

# Save evaluation results to a CSV file
evaluation_results = pd.DataFrame({
    'Training MSE': [train_mse],
    'Test MSE': [test_mse],
    'Training Accuracy': [train_accuracy],
    'Test Accuracy': [test_accuracy]
})
evaluation_results.to_csv(os.path.join(output_dir, 'brb_evaluation_results_POS1.csv'), index=False)

# Print optimized parameters
print("\nFinal Optimized Parameters:")
print("Belief Degrees:")
print(final_optimized_parameters_pso[:27].reshape((9, 3)))
print("Attribute Weights:")
print(final_optimized_parameters_pso[27:29])
print("Rule Weights:")
print(final_optimized_parameters_pso[29:38])

# Plot the results
plt.figure()
plt.plot(train['Group'].values, label='Original Group')
plt.plot(train_predicted, label='Predicted Group')
plt.title('Training Data: Original vs Predicted Group')
plt.xlabel('Sample Index')
plt.ylabel('Group')
plt.legend()
plt.show()

plt.figure()
plt.plot(test['Group'].values, label='Original Group')
plt.plot(test_predicted, label='Predicted Group')
plt.title('Test Data: Original vs Predicted Group')
plt.xlabel('Sample Index')
plt.ylabel('Group')
plt.legend()
plt.show()

# Save rule weights and belief degrees history to a CSV file
weights_degrees_history = []
for i, parameters in enumerate(rule_weights_beliefs_history):
    belief_degrees = parameters[:27].reshape((9, 3))
    rule_weights = parameters[29:38]
    attribute_weights = parameters[27:29]
    for j, (belief_degree, rule_weight) in enumerate(zip(belief_degrees, rule_weights)):
        weights_degrees_history.append({
            'Iteration': i,
            'Rule': j,
            'Belief Degree': belief_degree.tolist(),
            'Rule Weight': rule_weight
        })
    weights_degrees_history.append({
        'Iteration': i,
        'Attribute Weights': attribute_weights.tolist()
    })

weights_degrees_df = pd.DataFrame(weights_degrees_history)
weights_degrees_df.to_csv(os.path.join(output_dir, 'rule_weights_beliefs_history.csv'), index=False)

print("Optimization and Evaluation Completed.")
