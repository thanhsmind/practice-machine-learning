from pandas import read_csv
from pandas import set_option

#  1. LOAD DATA
filename = 'pima-indians-diabetes.data.csv'
names=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# Load data from link
#url = ' https://goo.gl/vhm1eU '
#data = read_csv(url, names=names)
data = read_csv(filename, names=names)

# 2. UNDERSTAND YOUR DATA WITH DESCRIPTIVE STATISTICS

# 2.1 PEEK AT YOUR DATA
print('PEEK AT YOUR DATA')
peek = data.head(20)
print(peek)

# 2.2 DIMENSIONS OF YOUR DATA
print('DIMENSIONS OF YOUR DATA')
print(data.shape)

# 2.3 DATA TYPE FOR EACH ATTRIBUTE
print('DATA TYPE FOR EACH ATTRIBUTE')
print(data.dtypes)

# 2.4 DESCRIPTIVE STATISTICS
print('DESCRIPTIVE STATISTICS')
set_option('display.width', 2)
# precision: đô chính xác, xác định bằng số lượng số 0 sau dấu .
set_option('precision', 3)

description = data.describe()
print(description)

#   2.5 CLASS DISTRIBUTION (CLASSIFICATION ONLY)
print('CLASS DISTRIBUTION (CLASSIFICATION ONLY)')
class_counts = data.groupby('class').size()
print(class_counts)

# 2.6 CORRELATIONS BETWEEN ATTRIBUTES
print('CORRELATIONS BETWEEN ATTRIBUTES')
correlations = data.corr(method='pearson')
print(correlations);