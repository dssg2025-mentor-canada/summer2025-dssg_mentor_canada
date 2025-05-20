import pandas as pd

df = pd.read_csv('/Users/isobeladams/Desktop/data.csv')

print(df.head())
print(df.info())
print(df.shape)

null_counts = df.isnull().sum()
null_counts = null_counts[null_counts > 0]
print(null_counts)
