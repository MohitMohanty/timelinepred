import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Load data from the Excel file
file_path = "../AI_TABLE.xls"
df = pd.read_excel(file_path)

# Select the first 7000 rows and fill missing values with "unassigned"
df = df.head(600)

# Define text feature columns and categorical columns
text_feature_columns = ["DEFECT_DESC", "JOBDETAIL", "JOBSUM", "JOBHEAD"]
categorical_columns = ["STR_SHIPNAME", "STR_REFIT_CODE", "FLG_OFFLOADED", "EQUIPMENT_NAME", "WI_QC_REMARK"]
target_column = "EMD"

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df[text_feature_columns + categorical_columns], df[target_column], test_size=0.2, random_state=500
)

# Create a text vectorizer for the text features
text_vectorizer = ColumnTransformer(
    transformers=[
        ('defect_text', CountVectorizer(), 'DEFECT_DESC'),
        ('jobdetail_text', CountVectorizer(), 'JOBDETAIL'),
        ('jobsum_text', CountVectorizer(), 'JOBSUM'),
        ('jobhead_text', CountVectorizer(), 'JOBHEAD')
    ],
    remainder='passthrough'
)

# Create a categorical encoder with handle_unknown='ignore'
categorical_encoder = OneHotEncoder(handle_unknown='ignore')

# Create a preprocessor pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_vectorizer, text_feature_columns),
        ('categorical', categorical_encoder, categorical_columns)
    ],
    remainder='passthrough'
)

# Create a RandomForestRegressor model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Define hyperparameters for grid search
param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

# Create grid search object
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Update the model with the best hyperparameters
model.set_params(**best_params)

# Fit the updated model to the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate RMSE using root_mean_squared_error
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Print the results
print(f'Best hyperparameters: {best_params}')
print(f'Root Mean Squared Error (RMSE) after tuning: {rmse:.2f}')
