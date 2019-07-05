from pandas import read_csv
from pandas import set_option
from matplotlib import pyplot
import numpy
from pandas.plotting import scatter_matrix

#  1. LOAD DATA
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin',
         'test', 'mass', 'pedi', 'age', 'class']
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
print(correlations)

# 2.7 Skew of Univariate Distributions. Độ nghiêng của mô hình phân bố
print('*')
print('===============Skew of Univariate Distributions==============')
print('*')

skew = data.skew()
print(skew)

# 3 Understand Your Data With Visualization

# 3.1 Univariate Plots
# 3.1.1 Histograms
print('*')
print('=============== Histograms ==============')
print('*')

data.hist()
# pyplot.show()
pyplot.savefig("histograms.png")

# 3.1.1 Density Plots
print('*')
print('=============== Density Plots ==============')
print('*')

data.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
# pyplot.show()
pyplot.savefig("density.png")


# 3.1.2 Box and Whisker Plots
print('*')
print('=============== Box and Whisker Plots ==============')
print('*')

data.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
# pyplot.show()
pyplot.savefig("box_whisker.png")

# 3.2 Multivariate Plots
# 3.2.1 Correlation Matrix Plot
print('*')
print('=============== Correlation Matrix Plot ==============')
print('*')

correlations = data.corr()
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0, 9, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)

# pyplot.show()
pyplot.savefig("Correlation_Matrix_Plot.png")

# 3.2.2  Scatter Plot Matrix
print('*')
print('===============  Scatter Plot Matrix ==============')
print('*')

scatter_matrix(data)
# pyplot.show()
pyplot.savefig("Scatter_Plot_Matrix.png")


# 4.1 Prepare Your Data For MachineLearning
print('*')
print('===============  Prepare Your Data For MachineLearning ==============')
print('*')


# 4.1.1 Rescale Data
print('*')
print('===============  Rescale Data ==============')
print('*')
from sklearn.preprocessing import MinMaxScaler
from numpy import set_printoptions

array = data.values
#separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(X)
# summarize transformed data
set_printoptions(precision=3)

print(rescaledX[0:5,:])

# 4.1.2 Standardize Data
print('*')
print('===============  Standardize Data ==============')
print('*')

from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
set_printoptions(precision=3)
print(rescaledX[0:5,:])

print(type(rescaledX))

# 4.1.3 Normalize Data
print('*')
print('===============  Normalize Data ==============')
print('*')

from sklearn.preprocessing import Normalizer

scaler= Normalizer().fit(X)
normalizedX = scaler.transform(X)
# summarize transformed data
set_printoptions(precision=3)
print(normalizedX[0:5,:])

# 4.1.3 Binarize Data (Make Binary)
print('*')
print('===============  Binarize Data (Make Binary) ==============')
print('*')
from sklearn.preprocessing import Binarizer

binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)
# summarize transformed data
set_printoptions(precision=3)
print(binaryX[0:5, :])

