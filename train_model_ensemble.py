import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the merged training data with BERT predictions
merged_train_data = pd.read_csv('merged_train_data_with_bert.csv')

# Ensure feature columns are included
feature_columns = [f'bert_feature_{i}' for i in range(768)] + \
                  ['url_length', 'num_dots', 'num_hyphens', 'num_slashes', 'num_underscores', 
                   'contains_gov', 'contains_com', 'contains_org']

# Prepare the data
X = merged_train_data[feature_columns].fillna(-1)
y = merged_train_data['phishy']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Save the scaler
joblib.dump(scaler, 'scaler_rf.pkl')
print("Scaler saved to 'scaler_rf.pkl'.")

# Define the RandomForestClassifier and hyperparameter grid
rf_model = RandomForestClassifier(random_state=42)

param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Perform RandomizedSearchCV to find the best hyperparameters
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_distributions, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Get the best estimator
best_rf_model = random_search.best_estimator_

# Evaluate the best model on the validation set
val_predictions = best_rf_model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
print(f'Validation Accuracy: {val_accuracy}')

# Save the best model
joblib.dump(best_rf_model, 'best_rf_model.pkl')
print("Best Random Forest model saved to 'best_rf_model.pkl'.")
