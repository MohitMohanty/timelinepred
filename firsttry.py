#code is written for a new excel file data prediction and plotting the graph
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

file_path = "../AI_TABLE.xls"
df = pd.read_excel(file_path)

df = df.head(600)
df.fillna("unassigned", inplace=True) 

# Assume you want to predict the "target_column" based on the "feature_columns" and "categorical_columns"
text_feature_columns = ["DEFECT_DESC", "JOBDETAIL", "JOBSUM", "JOBHEAD"]
categorical_columns = ["STR_SHIPNAME", "STR_REFIT_CODE", "FLG_OFFLOADED", "EQUIPMENT_NAME", "WI_QC_REMARK"]
target_column = "EMD"

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df[text_feature_columns + categorical_columns], df[target_column], test_size=0.2, random_state=42
)

# Use CountVectorizer to convert text data to numeric format for text columns
text_vectorizer = ColumnTransformer(
    transformers=[
        ('defect_text', CountVectorizer(), 'DEFECT_DESC'),
        ('jobdetail_text', CountVectorizer(), 'JOBDETAIL'),
        ('jobsum_text', CountVectorizer(), 'JOBSUM'),
        ('jobhead_text', CountVectorizer(), 'JOBHEAD')
    ],
    remainder='passthrough'
)

categorical_encoder = OneHotEncoder()

# Combine text and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_vectorizer, text_feature_columns),
        ('categorical', categorical_encoder, categorical_columns)
    ],
    remainder='passthrough'
)

# Create a pipeline with the preprocessor and the RandomForestRegressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])



model.fit(X_train, y_train)


y_predictions = model.predict(X_test)


print('R-squared error: ' + "{:.2%}".format(model.score(X_test, y_test)))
print('Root Mean Squared Error: ' + "{:.2f}".format(mean_squared_error(y_test, y_predictions, squared=False)))


