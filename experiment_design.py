import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime, timedelta
import joblib

# Purpose: This script will follow the steps of designing an experiment, 
#          running it, and analyzing the results using A/B testing principles.


# Load the dataset
data = pd.read_csv('data/user/user_churn.csv')

# Convert 'last_played' to datetime
data['last_played'] = pd.to_datetime(data['last_played'])

# Define features and target for user engagement
features = ['user_age', 'play_count']
target = 'churn'

X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the best model
best_rf = joblib.load('models/churn_prediction_model.pkl')

# Predict probabilities on the test set
y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

# Function to find optimal threshold
def find_optimal_threshold(fpr, tpr, thresholds):
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Find the optimal threshold
optimal_threshold = find_optimal_threshold(fpr, tpr, thresholds)

# Design A/B test
def design_ab_test(data, intervention_func, control_func, test_size=0.5):
    # Randomly assign users to control or intervention group
    data['group'] = np.where(np.random.rand(len(data)) < test_size, 'intervention', 'control')
    
    # Apply functions
    data.loc[data['group'] == 'intervention', 'result'] = data[data['group'] == 'intervention'].apply(intervention_func, axis=1)
    data.loc[data['group'] == 'control', 'result'] = data[data['group'] == 'control'].apply(control_func, axis=1)
    
    return data

# Example intervention function
def intervention_func(row):
    # Simulate an intervention, e.g., a targeted message to prevent churn
    if random.random() > 0.5:
        return 0  # User did not churn
    else:
        return 1  # User churned

# Example control function
def control_func(row):
    # Simulate the control condition, e.g., no message
    if random.random() > 0.7:
        return 0  # User did not churn
    else:
        return 1  # User churned

# Run A/B test
ab_test_results = design_ab_test(data, intervention_func, control_func)

# Evaluate A/B test results
intervention_group = ab_test_results[ab_test_results['group'] == 'intervention']
control_group = ab_test_results[ab_test_results['group'] == 'control']

# Calculate conversion rates
intervention_cr = intervention_group['result'].mean()
control_cr = control_group['result'].mean()

# Statistical significance test
from scipy.stats import chi2_contingency

contingency_table = pd.crosstab(ab_test_results['group'], ab_test_results['result'])
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.scatter(fpr[np.argmax(tpr - fpr)], tpr[np.argmax(tpr - fpr)], marker='o', color='black', label=f'Optimal threshold = {optimal_threshold:.2f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Print A/B test results
print(f'Intervention Conversion Rate: {intervention_cr:.2f}')
print(f'Control Conversion Rate: {control_cr:.2f}')
print(f'Chi-Squared Test p-value: {p:.4f}')

# Save A/B test results
ab_test_results.to_csv('data/ab_test_results.csv', index=False)
