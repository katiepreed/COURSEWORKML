"""
Linear Regression models (Ridge, Lasso, ElasticNet), use 5-fold cross-validation against mean R^2.
"""
import numpy as np

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from preprocess import preprocess_data

np.random.seed(123)

x_train, y_train, x_test = preprocess_data()

params_ridge = {'alpha': [0.01, 0.1, 1, 10, 100]}
params_lasso = {'alpha': [0.01, 0.1, 1, 10, 100]}
params_elastic = {
    'alpha': [0.01, 0.1, 1, 10, 100],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

"""
Ridge: R^2 = 0.2857087454680897, params = {'alpha': 100}
Lasso: R^2 = 0.28654311195837856, params = {'alpha': 0.1}
ElasticNet: R^2 = 0.28642262546539043, params = {'alpha': 0.1, 'l1_ratio': 0.9}
"""
for name, model, params in [
    ('Ridge', Ridge(), params_ridge),
    ('Lasso', Lasso(random_state=123), params_lasso),
    ('ElasticNet', ElasticNet(random_state=123), params_elastic),
]:
    grid = GridSearchCV(model, params, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(x_train, y_train)
    print(f"{name}: R^2 = {grid.best_score_}, params = {grid.best_params_}")