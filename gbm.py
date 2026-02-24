"""
Gradient Boosting Machine model, uses 5-fold cross-validation against mean R^2.
"""
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from preprocess import preprocess_data
import matplotlib.pyplot as plt

np.random.seed(123)

x_train, y_train, x_test = preprocess_data()

params = {
    'n_estimators': [300, 500, 800],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'min_samples_leaf': [5, 10, 20],
}

random_search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=123),
    params,
    n_iter=50,  # tries 50 random combinations
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=123
)

random_search.fit(x_train, y_train)
model = random_search.best_estimator_

"""
R^2: 0.4754682337706999
Best parameters: {'subsample': 0.7, 'n_estimators': 800, 'min_samples_leaf': 20, 'max_depth': 3, 'learning_rate': 0.01}
"""
print(f"R^2: {random_search.best_score_}") 
print(f"Best parameters: {random_search.best_params_}")

importance = pd.Series(model.feature_importances_, index=x_train.columns) # type: ignore
importance.sort_values(ascending=False).plot(kind='bar', title='Feature Importance')
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')

predictions = model.predict(x_test) # type: ignore

output = pd.DataFrame({'yhat': predictions})
output.to_csv('CW1_submission_K21008618.csv', index=False)