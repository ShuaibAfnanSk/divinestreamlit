import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("properties.csv")

df = df[["price","bedrooms","bathrooms","sqft_living","sqft_lot","city","statezip","sqft_basement","sqft_above","view","floors","condition","waterfront","yr_built","yr_renovated"]]

class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in self.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        return X

X = df.drop(["price"], axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

label_encoder = ['city', 'statezip']
scaler = [col for col in X.columns if col not in label_encoder]

preprocessor = ColumnTransformer(
    transformers=[
        ('label_encode', MultiColumnLabelEncoder(columns=label_encoder), label_encoder),
        ('scale', StandardScaler(), scaler)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

pipeline.fit(X_train, y_train)

bedrooms = st.number_input('Bedrooms', min_value=1, max_value=5, value=int(df['bedrooms'].mode()))
bathrooms = st.number_input('Bathrooms', min_value=1, max_value=5, value=int(df['bathrooms'].mode()))
floors = st.selectbox('Floors', [1], index=0)
sqft_living = st.slider('Square Feet Living', min_value=int(df['sqft_living'].min()), max_value=int(df['sqft_living'].max()), value=int(df['sqft_living'].mean()))
sqft_lot = st.slider('Square Feet Lot', min_value=int(df['sqft_lot'].min()), max_value=int(df['sqft_lot'].max()), value=int(df['sqft_lot'].mean()))
sqft_basement = st.slider('Square Feet Basement', min_value=int(df['sqft_basement'].min()), max_value=int(df['sqft_basement'].max()), value=int(df['sqft_basement'].median()))
sqft_above = st.slider('Square Feet Above', min_value=int(df['sqft_above'].min()), max_value=int(df['sqft_above'].max()), value=int(df['sqft_above'].mean()))
unique_cities = df['city'].unique()
city = st.selectbox('City', unique_cities, index=0)
unique_statezips = df['statezip'].unique()
statezip = st.selectbox('State ZIP', unique_statezips, index=0)
waterfront = st.selectbox('Waterfront', [0, 1], index=0)
view = st.selectbox('View', [1, 2, 3, 4, 5], index=0)
condition = st.selectbox('Condition', [1, 2, 3, 4, 5], index=0)
yr_built = st.slider('Year Built', min_value=int(df['yr_built'].min()), max_value=int(df['yr_built'].max()), value=int(df['yr_built'].median()))
yr_renovated = st.slider('Year Renovated', min_value=1990, max_value=int(df['yr_renovated'].max()), value=int(df['yr_renovated'].median()))



if st.button('Predict'):
    input_data = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'sqft_living': [sqft_living],
        'sqft_lot': [sqft_lot],
        'sqft_basement': [sqft_basement],
        'sqft_above': [sqft_above],
        'city': [city],
        'statezip': [statezip],
        'view': [view],
        'floors': [floors],
        'condition': [condition],
        'waterfront': [waterfront],
        'yr_built': [yr_built],
        'yr_renovated': [yr_renovated],
    })

    prediction = pipeline.predict(input_data)[0]
    st.success(f'Predicted Price: ${prediction:,.2f}')
