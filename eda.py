import pandas as pd

df = pd.read_csv('../../dssg-2025-mentor-canada/Data/Data_2020-Youth-Survey.csv')

print(df.head())
print(df.info())
print(df.shape)

null_counts = df.isnull().sum()
null_counts = null_counts[null_counts > 0]
print(null_counts)
