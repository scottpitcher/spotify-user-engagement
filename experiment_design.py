import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.api import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Load the dataset
data = pd.read_csv('data/user_interactions_with_churn.csv')

# Convert 'last_played' to datetime and create 'days_since_last_played'
data['last_played'] = pd.to_datetime(data['last_played'])
data['days_since_last_played'] = (pd.to_datetime('now') - data['last_played']).dt.days

# Determine churn on a per-user basis
user_last_played = data.groupby('user_id')['days_since_last_played'].max().reset_index()
user_last_played['churn'] = user_last_played['days_since_last_played'] > 20

# Add randomness to the churn
np.random.seed(42)
flip_prob = 0.15
random_flips = np.random.rand(len(user_last_played)) < flip_prob
user_last_played['churn'] = user_last_played['churn'] ^ random_flips

# Merge the churn information back into the original dataset
data = data.merge(user_last_played[['user_id', 'churn']], on='user_id', how='left')
data['churn'] = data['churn'].astype(int)

# Propensity Score Matching
def propensity_score_matching(data, treatment_col, outcome_col, covariates):
    # Train a model to estimate propensity scores
    X = data[covariates]
    y = data[treatment_col]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    data['propensity_score'] = model.predict_proba(X)[:, 1]

    # Match treated and untreated samples based on propensity scores
    treated = data[data[treatment_col] == 1]
    control = data[data[treatment_col] == 0]

    matched_control = control.iloc[(np.abs(control['propensity_score'].values[:, None] - treated['propensity_score'].values)).argmin(axis=0)]
    matched_data = pd.concat([treated, matched_control])

    return matched_data

# Define covariates for matching
covariates = ['user_age', 'play_count', 'days_since_last_played']
matched_data = propensity_score_matching(data, 'churn', 'outcome_variable', covariates)

# Perform causal inference analysis
X_matched = matched_data[covariates]
y_matched = matched_data['outcome_variable']

# Adding a constant term for OLS regression
X_matched = sm.add_constant(X_matched)
model = sm.OLS(y_matched, X_matched).fit()

# Print the summary of the model
print(model.summary())

# Plotting the ROC curve for the propensity score model
fpr, tpr, thresholds = roc_curve(data['churn'], data['propensity_score'])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Save the matched data
matched_data.to_csv('data/matched_user_data.csv', index=False)
