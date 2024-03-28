import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Load data from the Excel file
file_path = "../BI_TABLE.xls"
df = pd.read_excel(file_path)

# Select the first 600 rows and fill missing values with "unassigned"
#df = df.head(600)
df.fillna("unassigned", inplace=True)

# Define text feature columns and categorical columns
text_feature_columns = ["DEFECT_DESC", "JOBDETAIL", "JOBSUM", "JOBHEAD"]
categorical_columns = ["STR_SHIPNAME", "STR_REFIT_CODE", "FLG_OFFLOADED", "EQUIPMENT_NAME", "WI_QC_REMARK"]
target_column = "EMD"

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df[text_feature_columns + categorical_columns], df[target_column], test_size=0.2, random_state=42
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

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate R-squared error
r_squared = model.score(X_test, y_test)

# Calculate RMSE
#rmse = mean_squared_error(y_test, y_pred, squared=False)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Print the results
print(f'R-squared error: {r_squared:.2%}')
print(f'Root Mean Squared Error: {rmse:.2f}')