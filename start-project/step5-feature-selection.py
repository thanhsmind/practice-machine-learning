# 5.1 Feature Selection For Machine Learning
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from numpy import set_printoptions
from pandas import read_csv
print('*')
print('===============  Feature Selection For Machine Learning ==============')
print('*')

# 5.1.2 Univariate selection
print('*')
print('===============  Univariate selection ==============')
print('*')

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin',
         'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

# Feature Extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5, :])

# 5.1.3 Recursive Feature Elimination
print('*')
print('===============  Recursive Feature Elimination ==============')

# feature extraction
model = LogisticRegression(solver="lbfgs", multi_class='auto')
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)

print(fit.n_features_)
print(fit.support_)
print(fit.ranking_)

# 5.1.4 Feature Extraction with PCA (Principal Component Analysis)
print('*')
print('===============  Feature Extraction with PCA ==============')
print('*')

# Feature Extraction 
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print(fit.explained_variance_ratio_)
print(fit.components_)

# 5.1.5 Feature Importance
print('*')
print('===============  Feature Importance ==============')
print('*')

from sklearn.ensemble import ExtraTreesClassifier

# Feature extraction
model = ExtraTreesClassifier()
model.fit(X,Y)

print(model.feature_importances_)