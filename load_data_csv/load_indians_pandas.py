from pandas import read_csv
from pandas import set_option
from matplotlib import pyplot


#  1. LOAD DATA
filename = 'pima-indians-diabetes.data.csv'
names=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# Load data from link
#url = ' https://goo.gl/vhm1eU '
#data = read_csv(url, names=names)
data = read_csv(filename, names=names)

# 2. UNDERSTAND YOUR DATA WITH DESCRIPTIVE STATISTICS

# 2.1 PEEK AT YOUR DATA
print('*')
print('==============PEEK AT YOUR DATA==============')
print('*')

peek = data.head(20)
print(peek)

# 2.2 DIMENSIONS OF YOUR DATA
print('*')
print('==============DIMENSIONS OF YOUR DATA==============')
print('*')

print(data.shape)

# 2.3 DATA TYPE FOR EACH ATTRIBUTE
print('*')
print('==============DATA TYPE FOR EACH ATTRIBUTE==============')
print('*')

print(data.dtypes)

# 2.4 DESCRIPTIVE STATISTICS
print('*')
print('==============DESCRIPTIVE STATISTICS==============')
print('*')

set_option('display.width', 2)
# precision: đô chính xác, xác định bằng số lượng số 0 sau dấu .
set_option('precision', 3)

description = data.describe()
print(description)

#   2.5 CLASS DISTRIBUTION (CLASSIFICATION ONLY)
print('*')
print('==============CLASS DISTRIBUTION (CLASSIFICATION ONLY)==============')
print('*')

class_counts = data.groupby('class').size()
print(class_counts)

# 2.6 CORRELATIONS BETWEEN ATTRIBUTES
print('*')
print('==============CORRELATIONS BETWEEN ATTRIBUTES==============')
print('*')

correlations = data.corr(method='pearson')
print(correlations);

# 2.7 Skew of Univariate Distributions. Độ nghiêng của mô hình phân bố 
print('*')
print('===============Skew of Univariate Distributions==============')
print('*')

skew = data.skew()
print(skew)

# 3 Understand Your Data With Visualization 

# 3.1 
# 3.1.1 Histograms
data.hist()
#pyplot.show()
pyplot.savefig("matplotlib.png")