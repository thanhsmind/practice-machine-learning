from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values


X = array[:, 0:8]
Y  = array[:,8]

# Train and Test sets
test_size = 0.33
seed = 7
# split dataset to two sets, 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=test_size, random_state=seed)

model = LogisticRegression()
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
print(X_train)
print(Y_train)
print("Accuracy: ")
print(result*100.0)

# Evaluate using Cross Validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

num_folds=10
seed = 7 
kfold = KFold(n_splits=num_folds, random_state=seed)
model = LogisticRegression()
results = cross_val_score(model, X,Y, cv=kfold)
print(results.mean()*100.0)


# Algorithm Evaluation Metrics

# Classification Metrics

# Classification Accuracy

print('Classification Accuracy')

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()

scoring = 'accuracy'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print('Accuracy: mean: ', results.mean(),'std: ', results.std() )


print('Logarithmic Loss')

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
scoring= 'neg_log_loss'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print('Loglos: {:0.3f} ({:0.3f})'.format(results.mean(),  results.std()) )

print('Classification Area Under ROC Curve (AUC)')

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
scoring = 'roc_auc'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print('AUC: {:0.3f} ({:0.3f})'.format(results.mean(), results.std()))


