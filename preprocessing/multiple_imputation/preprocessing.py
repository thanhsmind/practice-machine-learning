from pandas import read_csv

#  1. LOAD DATA
filename = 'data.csv'
names = ['Item', 'Y', 'X']

data = read_csv(filename, names=names)

print(data.shape)
print(data)
print(data.describe())

# Multiple Imputation  by chained equations
from fancyimpute import MICE

