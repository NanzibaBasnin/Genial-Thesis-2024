import os
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from pyswarm import pso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Constants for Alzheimer's diagnosis
AD_SCORE = 1.0
MCI_SCORE = 0.7
CN_SCORE = 0.2

# Load the CSV file
file_path = 'C:\\Users\\HP\\Documents\\Discertation\\Code\\Fed_BRB\\selected_columns.csv'

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
client_train_data = np.array_split(train, 3)
client_test_data = np.array_split(test, 3)

# Scaling the features
scaler = MinMaxScaler()
for i in range(3):
    client_train_data[i][['Crisp_Value', 'Age']] = scaler.fit_transform(client_train_data[i][['Crisp_Value', 'Age']])
    client_test_data[i][['Crisp_Value', 'Age']] = scaler.transform(client_test_data[i][['Crisp_Value', 'Age']])

# Save split data into CSV files
output_dir = 'C:\\Users\\HP\\Documents\\Discertation\\Code\\Fed_BRB\\output'
os.makedirs(output_dir, exist_ok=True)

for i in range(3):
    client_train_data[i].to_csv(os.path.join(output_dir, f'client_{i+1}_train_set.csv'), index=False)
    client_test_data[i].to_csv(os.path.join(output_dir, f'client_{i+1}_test_set.csv'), index=False)

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
        sum_belief = np.sum(parameters[i*3:(i+1)*3])
        if sum_belief == 0:
            parameters[i*3:(i+1)*3] = [1/3, 1/3, 1/3]  # Assign equal belief if sum is zero
        else:
            parameters[i*3:(i+1)*3] /= sum_belief
    return parameters

# Define the evaluation model function
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

# Calculate accuracy
def calculate_accuracy(original, predicted, threshold=0.1):
    correct_predictions = np.abs(np.array(original) - np.array(predicted)) <= threshold
    return np.mean(correct_predictions)

# Function to run PSO for a client
def run_pso_for_client(client_idx, client_data):
    print(f"Starting PSO for Client {client_idx + 1}...")
    start_time = time.time()
    optimized_parameters_pso, _ = pso(lambda x: objective_function(x, client_data[client_idx]), lb, ub, swarmsize=50, maxiter=100)
    optimized_parameters_pso = ensure_belief_degrees(optimized_parameters_pso)
    print(f"PSO for Client {client_idx + 1} completed in {time.time() - start_time} seconds.")
    return optimized_parameters_pso

# Iterative training process
num_iterations = 10
for iteration in range(num_iterations):
    print(f"\nIteration {iteration + 1}/{num_iterations}")
    optimized_parameters_clients = []
    for i in range(3):
        optimized_parameters_pso = run_pso_for_client(i, client_train_data)
        optimized_parameters_clients.append(optimized_parameters_pso)
    
        # Print optimized parameters for each client
        print(f"\nOptimized Parameters for Client {i+1}:")
        print("Belief Degrees:")
        belief_degrees = optimized_parameters_pso[:27].reshape((9, 3))
        print(belief_degrees)
        print("Attribute Weights:")
        attribute_weights = optimized_parameters_pso[27:29]
        print(attribute_weights)
        print("Rule Weights:")
        rule_weights = optimized_parameters_pso[29:38]
        print(rule_weights)

        # Evaluate on client train and test sets
        client_train_mse, client_train_predicted = evaluate_model(optimized_parameters_pso, client_train_data[i])
        client_test_mse, client_test_predicted = evaluate_model(optimized_parameters_pso, client_test_data[i])

        client_train_accuracy = calculate_accuracy(client_train_data[i]['Group'], client_train_predicted)
        client_test_accuracy = calculate_accuracy(client_test_data[i]['Group'], client_test_predicted)

        print(f"Client {i+1} Training MSE: {client_train_mse}, Training Accuracy: {client_train_accuracy}")
        print(f"Client {i+1} Testing MSE: {client_test_mse}, Testing Accuracy: {client_test_accuracy}")

        # Save client results
        client_train_results = pd.DataFrame({'Observed': client_train_data[i]['Group'], 'Predicted': client_train_predicted})
        client_test_results = pd.DataFrame({'Observed': client_test_data[i]['Group'], 'Predicted': client_test_predicted})

        client_train_results.to_csv(os.path.join(output_dir, f'client_{i+1}_train_results_iteration_{iteration + 1}.csv'), index=False)
        client_test_results.to_csv(os.path.join(output_dir, f'client_{i+1}_test_results_iteration_{iteration + 1}.csv'), index=False)

    # Average the optimized parameters
    averaged_parameters = np.mean(optimized_parameters_clients, axis=0)
    averaged_parameters = ensure_belief_degrees(averaged_parameters)

    # Run PSO again on the averaged parameters to get the final optimized parameters
    print("Starting final PSO on averaged parameters for next iteration...")
    start_time = time.time()
    final_optimized_parameters_pso, _ = pso(lambda x: objective_function(x, train), lb, ub, swarmsize=50, maxiter=100)
    final_optimized_parameters_pso = ensure_belief_degrees(final_optimized_parameters_pso)
    print(f"Final PSO for next iteration completed in {time.time() - start_time} seconds.")

    # Print and save the final optimized parameters
    print("\nFinal Optimized Parameters after Averaging:")
    print("Belief Degrees:")
    final_belief_degrees = final_optimized_parameters_pso[:27].reshape((9, 3))
    print(final_belief_degrees)
    print("Attribute Weights:")
    final_attribute_weights = final_optimized_parameters_pso[27:29]
    print(final_attribute_weights)
    print("Rule Weights:")
    final_rule_weights = final_optimized_parameters_pso[29:38]
    print(final_rule_weights)

    belief_degrees = final_optimized_parameters_pso[:27].reshape((9, 3)).tolist()
    attribute_weights = final_optimized_parameters_pso[27:29].tolist()
    rule_weights = final_optimized_parameters_pso[29:38].tolist()

    final_params_df = pd.DataFrame({
        'Belief Degrees': [belief_degrees],
        'Attribute Weights': [attribute_weights],
        'Rule Weights': [rule_weights]
    })
    final_params_df.to_csv(os.path.join(output_dir, f'final_optimized_parameters_iteration_{iteration + 1}.csv'), index=False)

    # Evaluate on the full training and test sets
    train_mse, train_predicted = evaluate_model(final_optimized_parameters_pso, train)
    test_mse, test_predicted = evaluate_model(final_optimized_parameters_pso, test)

    train_accuracy = calculate_accuracy(train['Group'], train_predicted)
    test_accuracy = calculate_accuracy(test['Group'], test_predicted)

    print(f"Iteration {iteration + 1} Final Training MSE: {train_mse}, Training Accuracy: {train_accuracy}")
    print(f"Iteration {iteration + 1} Final Testing MSE: {test_mse}, Testing Accuracy: {test_accuracy}")

    # Save the final predicted and observed values to CSV files
    train_results = pd.DataFrame({'Observed': train['Group'], 'Predicted': train_predicted})
    test_results = pd.DataFrame({'Observed': test['Group'], 'Predicted': test_predicted})

    train_results.to_csv(os.path.join(output_dir, f'final_train_results_iteration_{iteration + 1}.csv'), index=False)
    test_results.to_csv(os.path.join(output_dir, f'final_test_results_iteration_{iteration + 1}.csv'), index=False)

    # Save evaluation results to a CSV file
    evaluation_results = pd.DataFrame({
        'Training MSE': [train_mse],
        'Test MSE': [test_mse],
        'Training Accuracy': [train_accuracy],
        'Test Accuracy': [test_accuracy]
    })
    evaluation_results.to_csv(os.path.join(output_dir, f'final_evaluation_results_iteration_{iteration + 1}.csv'), index=False)

print("Optimization and Evaluation Completed.")
