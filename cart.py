"""
CART (Decision Tree) model, uses 5-fold cross-validation against mean R^2.
"""
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import  GridSearchCV
from preprocess import preprocess_data

np.random.seed(123)

x_train, y_train, x_test = preprocess_data()

params = {
    'max_depth': [3, 5, 7, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5, 10],
}

grid_search = GridSearchCV(
    DecisionTreeRegressor(random_state=123),
    params,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(x_train, y_train)

"""
R^2: 0.42241011566837317
Best parameters: {'max_depth': 5, 'min_samples_leaf': 5, 'min_samples_split': 2}
"""
print(f"R^2: {grid_search.best_score_}")
print(f"Best parameters: {grid_search.best_params_}")