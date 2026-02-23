"""
Exploratory Data Analysis for the training dataset. 

Analysis includes:

- dataset shape, missing values, and duplicates 
- skewness of numeric features and invalid rows
- pearson correlations with outcome variable and multicollinearity

All plots have been saved to plots/ directory. 
"""
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
import seaborn as sns
import numpy as np

train_data = pd.read_csv('CW1_train.csv')

numeric_columns = train_data.select_dtypes(include='number').columns.drop('outcome') # not inc
categorical_columns = train_data.select_dtypes(include=['object', 'category', 'str']).columns

# ---------------- ANALYSIS OF DATASET ---------------------------------------------

train_data_shape = train_data.shape # (10000, 31): 10 000 rows and 31 columns 
num_numeric_features = len(numeric_columns) # 27 numeric features
num_categorical_features = len(categorical_columns) # 3 categorical features
num_missing_values = train_data.isnull().sum().sum()  # 0 missing values
num_duplicates = train_data.duplicated().sum() # 0 duplicates

# --------------- SKEWNESS of NUMERIC FEATURES --------------------------------------

# show skewness for all numeric features 
numeric_skewness = train_data[numeric_columns].skew().sort_values(key=abs, ascending=False) # type: ignore
# get the features that are skewed
skewed_numeric_features = numeric_skewness[numeric_skewness.abs() > 0.1].index
# get the features that are not skewed
non_skewed_numeric = numeric_columns.drop(skewed_numeric_features)

# --------------- INVALID ROWS ------------------------------------------------------

# rows with zero dimensions 
zero_mask = (train_data['x'] == 0) | (train_data['y'] == 0) | (train_data['z'] == 0)
num_zero_dimension_rows = zero_mask.sum() #  4

# extreme y-values 
y_extremes = (train_data['y'] > 30).sum() #  1

total_invalid_rows = num_zero_dimension_rows + y_extremes

# --------------- CORRELATION WITH OUTPUT -------------------------------------------

# one-hot encode categorical features 
train_onehot = pd.get_dummies(train_data, columns=categorical_columns, drop_first=False) # type: ignore

# get all features excluding outcome
all_features = train_onehot.columns.drop('outcome')

# calculate Pearson correlation between each feature and outcome
correlations = train_onehot[all_features].corrwith(train_onehot['outcome']).sort_values(key=abs, ascending=False)

# --------------- MULTICOLLINEARITY ----------------------------------------------------

feature_col = train_data[skewed_numeric_features].corr()

high_correlation_pairs = []

for i in range(len(feature_col.columns)):
    for j in range(i +1, len(feature_col.columns)):
        num = feature_col.iloc[i, j]
        if abs(num) > 0.7: 
            high_correlation_pairs.append((feature_col.columns[i], feature_col.columns[j], round(num, 4)))


# -------------- BOX PLOTS : NUMERIC ---------------------------------------------------

fig, ax = plt.subplots(1, len(skewed_numeric_features), figsize=(15, 5))
for i, col in enumerate(skewed_numeric_features):
    train_data[[col]].boxplot(ax=ax[i])

    q1 = train_data[col].quantile(0.25)
    median = train_data[col].median()
    q3 = train_data[col].quantile(0.75)

    min_val = train_data[col].min()
    max_val = train_data[col].max()

    texts = []
    for label, val in [('Min', min_val),('Q1', q1), ('Med', median), ('Q3', q3), ('Max', max_val)]:
        texts.append(ax[i].text(1.3, val, f'{label}: {val:.0f}', fontsize=7))
 
    adjust_text(texts, ax=ax[i])
 

plt.tight_layout()
fig.savefig("plots/boxplots.png", dpi=300)

# --------------- HISTOGRAM : OUTCOME -----------------------------------------------

fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(train_data['outcome'], bins=40, edgecolor='black', color="#70dc80")
ax.set_title(f'Outcome')

plt.tight_layout()
fig.savefig("plots/outcome.png", dpi=300)

# --------------- HISTOGRAM : SKEWED  ----------------------------------------------

fig, ax = plt.subplots(1, len(skewed_numeric_features), figsize=(18, 5))
for i, col in enumerate(skewed_numeric_features):
    ax[i].hist(train_data[col], bins=40, edgecolor='black', color="#70dc80")
    ax[i].set_title(f'{col} (skew={train_data[col].skew():.2f})')

plt.tight_layout()
fig.savefig("plots/skewed_histograms.png", dpi=300)

# -------------- HISTOGRAM : NON-SKEWED --------------------------------------------

n_non_skewed = len(non_skewed_numeric)
n_cols = 5
n_rows = (n_non_skewed + n_cols - 1) // n_cols

fig, ax = plt.subplots(4, 5, figsize=(16, 12))
ax = ax.flatten()

for i, col in enumerate(non_skewed_numeric):
    ax[i].hist(train_data[col], bins=40, edgecolor='black', color="#70dc80")
    ax[i].set_title(f'{col} (skew={train_data[col].skew():.2f})')
    ax[i].set_xlabel('')

for j in range(n_non_skewed, len(ax)):
    ax[j].set_visible(False)

fig.tight_layout()
fig.savefig("plots/nonskewed_histograms.png", dpi=300, bbox_inches="tight")

# -------------- BAR CHART: CATEGORICAL  -------------------------------------

fig, ax = plt.subplots(1, len(categorical_columns), figsize=(15, 5))
for i, col in enumerate(categorical_columns):
    train_data[col].value_counts().plot(kind='bar', ax=ax[i], edgecolor='black', color="#70dc80")

plt.tight_layout()
fig.savefig("plots/categorical_barchart.png", dpi=300, bbox_inches="tight")

# -------------- BAR CHART: OUTCOME CORRELATION -------------------------------

fig, ax = plt.subplots(figsize=(16, 6))

correlations.plot(kind='bar', ax=ax, edgecolor='black', color="#70dc80")
ax.set_title('Feature Correlation with Outcome')
ax.set_ylabel('Pearson Correlation')

fig.savefig("plots/correlation_outcome_barchart.png", dpi=300, bbox_inches="tight")

# ---------------- HEATMAP: MULTICOLLINEARITY SUBSET ----------------------------

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(feature_col, dtype=bool))
sns.heatmap(feature_col, mask=mask, annot=True, cmap='coolwarm', center=0, ax=ax)
fig.savefig("plots/correlation_heatmap.png", dpi=300, bbox_inches="tight")


# ---------------- HEATMAP: MULTICOLLINEARITY ALL FEATURES ----------------------

fig, ax = plt.subplots(figsize=(16, 12))
corr_matrix = train_onehot[all_features].corr() # type: ignore
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=ax)
fig.tight_layout()
fig.savefig("plots/correlation_heatmap_all.png", dpi=300, bbox_inches="tight")

# --------------------------------------------------------------------------------

print(f"\nShape of data: {train_data_shape}")
print(f"\nNumber of numeric features: {num_numeric_features}")
print(f"\nNumber of categorical features: {num_categorical_features}")
print(f"\nNumber of missing values: {num_missing_values}")
print(f"\nNumber of duplicates: {num_duplicates}")

print(f"\nSkewness of numeric features:")
print(numeric_skewness)

print(f"\nNumber of rows with zero dimensions for x,y,z: {num_zero_dimension_rows}")
print(f"\nNumber extreme y-values: {y_extremes}")
print(f"\nNumber of invalid rows: {total_invalid_rows}")

print(f"\nCorrelations with outcome: ")
print(correlations)

print(f"\nHigh Correlation feature pairs: ")
for feature1, feature2, correlation in high_correlation_pairs:
    print(f"{feature1}, {feature2}: {correlation}")
