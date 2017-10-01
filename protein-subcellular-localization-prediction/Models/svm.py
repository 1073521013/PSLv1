from __future__ import division
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time
import itertools
from scipy.stats import randint as sp_randint
from sklearn.feature_selection import SelectKBest , f_classif
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
def accuracy(pred, actual):
    """Returns percentage of correctly classified labels"""
    return sum(pred==actual) / len(actual)


	# Import data and split out labels
train = pd.read_csv('../datasets/dataset_iloc/train.csv')
test = pd.read_csv('../datasets/dataset_iloc/test.csv')

	# Split into labels, names and data
y_train = train['class']
names_train = train['name']
X_train = train.drop(['class', 'name', 'sequence'], axis=1)

y_test = test['class']
names_test = test['name']
X_test = test.drop(['class', 'name', 'sequence'], axis=1)
std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.fit_transform(X_test)
X_train = SelectKBest(f_classif,k=70).fit_transform(X_train,y_train)
X_test = SelectKBest(f_classif,k=70).fit_transform(X_test,y_test)

from sklearn.svm import SVC    
#model = SVC(kernel='rbf', probability=True)    
#param_grid = {'C': [ 1,5,10, 15,20,25], 'gamma': [ 0.1,0.01,0.05,0.005]}    
#grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)    
#grid_search.fit(X_train, y_train)    
#best_parameters = grid_search.best_estimator_.get_params()   
#model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)    
model = SVC(C=5,gamma=0.01)
model.fit(X_train, y_train)    
 # Print train and test accuracy
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
print("Training Accuracy = %f" % accuracy(y_train_pred, y_train))
print("Test Accuracy = %f" % accuracy(y_test_pred, y_test))

cf_val=confusion_matrix(np.array(y_test), np.array(y_test_pred))

# Plot confusion matrix 
plt.figure(figsize=(8,8))
cmap=plt.cm.Blues   
plt.imshow(cf_val, interpolation='nearest', cmap=cmap)
plt.title('Confusion matrix validation set')
plt.colorbar()
tick_marks = np.arange(11)
classes =['cytoplasmic', 'extracellular', 'nuclear', 'plasma_membrane',"ER","lysosomal",                  
          "Golgi","chloroplast","peroxisomal","mitochondrial","vacuolar"]

plt.xticks(tick_marks, classes, rotation=60)
plt.yticks(tick_marks, classes)

thresh = cf_val.max() / 2.
for i, j in itertools.product(range(cf_val.shape[0]), range(cf_val.shape[1])):
    plt.text(j, i, cf_val[i, j],
             horizontalalignment="center",
             color="white" if cf_val[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True location')
plt.xlabel('Predicted location');
plt.savefig('svm.png')

