import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your dataset
df = pd.read_csv('diabetes.csv')  # Replace 'diabetes.csv' with the name of your dataset file

# Preprocess your dataset as needed
# Example: Handle missing values, encode categorical variables, etc.

# Separate features and target variable
x = df.drop(columns=['diabetes'])  # Replace 'target_column' with the name of your target column
y = df['diabetes']

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Assuming 'x' is your input features DataFrame
categorical_columns = ['gender', 'smoking_history']  # Add other categorical columns if needed

# Use ColumnTransformer to apply one-hot encoding to categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns)
    ],
    remainder='passthrough'
)

x_encoded = preprocessor.fit_transform(x)

from sklearn.feature_selection import SelectKBest, f_classif

# Assuming 'x_encoded' is your one-hot encoded features DataFrame
# Assuming 'y' is your target variable

# Use SelectKBest with f_classif score function on x_encoded
k_best = 5  # You can choose any value based on your requirements
feat_selector = SelectKBest(score_func=f_classif, k=k_best)
x_encoded_selected = feat_selector.fit_transform(x_encoded, y)

# Assuming you want to use the features selected by f_classif
selected_features = pd.DataFrame(x_encoded_selected, columns=feat_selector.get_support(indices=True))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(selected_features, y, test_size=0.3, random_state=42)

# Define the parameter grid for RandomizedSearchCV
param_grid = {
    'n_estimators': [50, 100],  # Number of trees in the forest
    'max_depth': [None, 10, 20],  # Maximum depth of the trees
    'min_samples_split': [2, 5],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2]  # Minimum number of samples required at each leaf node
}

# Create a Random Forest classifier object
rf = RandomForestClassifier()

# Create RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)

# Perform random search to find the best hyperparameters
random_search.fit(X_train, y_train)

# Get the best estimator from the random search
best_rf = random_search.best_estimator_

# Print the best hyperparameters found
print("Best hyperparameters:", random_search.best_params_)

# Print the accuracy on the training set
print("Training Accuracy:", best_rf.score(X_train, y_train))

# Save the trained model
pickle.dump(best_rf, open('model.pkl', 'wb'))
