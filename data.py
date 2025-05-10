import pandas as pd

data = pd.read_csv('Data/Final.csv')

print(data.info())

print(data.describe())

print(data.head())


