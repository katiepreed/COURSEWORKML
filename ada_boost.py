"""
AdaBoost model, uses 5-fold cross-validation against mean R^2.
"""
import numpy as np

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from preprocess import preprocess_data

np.random.seed(123)

x_train, y_train, x_test = preprocess_data()

params = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
    'estimator__max_depth': [1, 3, 5, 7],
}

grid_search = GridSearchCV(
    AdaBoostRegressor(estimator=DecisionTreeRegressor(random_state=123), random_state=123),
    params,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(x_train, y_train)

"""
R^2: 0.461505707218094
Best parameters: {'estimator__max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 500}
"""
print(f"R^2: {grid_search.best_score_}")
print(f"Best parameters: {grid_search.best_params_}")