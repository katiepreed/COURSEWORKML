"""
Shared preprocessing pipeline for all models:

- removes imvalid rows
- drops multicollinear features 
- one-hot encodes the categorical columns
- aligns the test columns to training columns
- applies scaling to all features 
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    train = pd.read_csv("CW1_train.csv")
    test = pd.read_csv("CW1_test.csv")

    # select rows where x, y, z are not zero
    # diamonds can't have zero dimensions
    x_y_z_mask = (train['x'] == 0) | (train['y'] == 0) | (train['z'] == 0)
    train = train[~x_y_z_mask].copy()

    # remove outliers in y
    outlier_mask = train['y'] > 30 
    train = train[~outlier_mask].copy()
 
    # drop multicollinear features (apart from carat)
    drop_cols = ['price', 'x', 'y', 'z']
    train = train.drop(columns=drop_cols)
    test = test.drop(columns=drop_cols)

    # one-hot encode categorical features
    categorical_cols = ['cut', 'color', 'clarity']
    train = pd.get_dummies(train, columns=categorical_cols, drop_first=True)
    test = pd.get_dummies(test, columns=categorical_cols, drop_first=True)

    # separate features and outcomes 
    y_train = train['outcome']
    x_train = train.drop(columns=['outcome'])

    # make test set have same columns as training set
    x_test = test.reindex(columns=x_train.columns, fill_value=0)

    features = x_train.columns.tolist()

    # standardise the features
    scaler = StandardScaler()
    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=features, index=x_train.index)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=features, index=x_test.index)

    return x_train, y_train, x_test





