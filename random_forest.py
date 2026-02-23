"""
Random Forest model, uses 5-fold cross-validation against mean R^2.
"""
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from preprocess import preprocess_data

np.random.seed(123)

x_train, y_train, x_test = preprocess_data()

params = {
    'n_estimators': [100, 300, 500], 
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10],
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=123, oob_score=True),
    params,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(x_train, y_train)
model = grid_search.best_estimator_

# R^2: 0.4594404434236875
print(f"R^2: {grid_search.best_score_}")
print(f"Best parameters: {grid_search.best_params_}")