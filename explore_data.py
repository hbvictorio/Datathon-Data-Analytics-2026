import pandas as pd
import numpy as np

# Load the PEDE dataset (it's a CSV with semicolon delimiter despite .xlsx extension)
df = pd.read_csv('/home/ubuntu/upload/PEDE_PASSOS_DATASET_FIAP.xlsx', sep=';', encoding='utf-8')
print('Shape:', df.shape)
print('\nColumns:')
for i, c in enumerate(df.columns):
    print(f'  [{i}] {c} -> dtype: {df[c].dtype}, nulls: {df[c].isnull().sum()}, non-null: {df[c].notna().sum()}')

print('\n\nDescribe numeric:')
print(df.describe().to_string())

print('\n\nFirst 3 rows:')
print(df.head(3).to_string())

# Save as proper CSV for later use
df.to_csv('/home/ubuntu/datathon_passos/data/PEDE_PASSOS_DATASET.csv', index=False)
print('\nSaved clean CSV to data/PEDE_PASSOS_DATASET.csv')
