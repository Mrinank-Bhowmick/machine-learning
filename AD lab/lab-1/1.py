# WAP to demonstrate the various data preprocessing steps in any ML model
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Sample dataset
data = {
    'Age': [25, np.nan, 35, 40, np.nan],
    'Salary': [50000, 60000, np.nan, 80000, 70000],
    'Country': ['France', 'Spain', 'Germany', 'France', 'Germany'],
    'Purchased': ['No', 'Yes', 'No', 'No', 'Yes']
}
df = pd.DataFrame(data)

print("Original Data:")
print(df)

# Handling missing data
# Impute missing numerical values with the mean
num_imputer = SimpleImputer(strategy='mean')
df[['Age', 'Salary']] = num_imputer.fit_transform(df[['Age', 'Salary']])

print("\nAfter Handling Missing Data:")
print(df)

# Encoding categorical data
# OneHotEncoder for 'Country', Label Encoding for 'Purchased'
column_transformer = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), ['Country'])
    ],
    remainder='passthrough'
)
df_encoded = column_transformer.fit_transform(df)

# Convert back to DataFrame for better visualization
encoded_columns = column_transformer.named_transformers_['encoder'].get_feature_names_out(['Country'])
df_encoded = pd.DataFrame(df_encoded, columns=list(encoded_columns) + ['Age', 'Salary', 'Purchased'])

print("\nAfter Encoding Categorical Data:")
print(df_encoded)

# Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_encoded.iloc[:, :-1])  # Exclude the 'Purchased' column for scaling

# Add the 'Purchased' column back
df_scaled = pd.DataFrame(scaled_features, columns=df_encoded.columns[:-1])
df_scaled['Purchased'] = df_encoded['Purchased']

print("\nAfter Feature Scaling:")
print(df_scaled)
