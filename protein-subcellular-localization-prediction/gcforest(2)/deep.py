# coding: utf-8

# In[1]:
import numpy as np
import random
import uuid
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest , f_classif
from deep_forest import MGCForest

# In[2]:


train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
train.loc[train["class"]=="Cell.membrane","class"]=0
train.loc[train["class"]=="Cytoplasm","class"]=1
train.loc[train["class"]=="Endoplasmic.reticulum","class"]=2
train.loc[train["class"]=="Golgi.apparatus","class"]=3
train.loc[train["class"]=="Lysosome/Vacuole","class"]=4
train.loc[train["class"]=="Mitochondrion","class"]=5
train.loc[train["class"]=="Nucleus","class"]=6
train.loc[train["class"]=="Peroxisome","class"]=7
train.loc[train["class"]=="Plastid","class"]=8
train.loc[train["class"]=="Extracellular","class"]=9


test.loc[test["class"]=="Cell.membrane","class"]=0
test.loc[test["class"]=="Cytoplasm","class"]=1
test.loc[test["class"]=="Endoplasmic.reticulum","class"]=2
test.loc[test["class"]=="Golgi.apparatus","class"]=3
test.loc[test["class"]=="Lysosome/Vacuole","class"]=4
test.loc[test["class"]=="Mitochondrion","class"]=5
test.loc[test["class"]=="Nucleus","class"]=6
test.loc[test["class"]=="Peroxisome","class"]=7
test.loc[test["class"]=="Plastid","class"]=8
test.loc[test["class"]=="Extracellular","class"]=9

# Split into labels, names and data
y_train = train['class']
names_train = train['name']
X_train = train.drop(['class', 'name', 'sequence'], axis=1)

y_test = test['class']
names_test = test['name']
X_test = test.drop(['class', 'name', 'sequence'], axis=1)


# In[3]:
#X_train = X_train.as_matrix
#X_test = X_test.as_matrix
std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.fit_transform(X_test)

y_train = np.asarray(y_train, dtype='int')
y_test = np.asarray(y_test, dtype='int')
X_train = SelectKBest(f_classif,k=70).fit_transform(X_train,y_train)
X_test = SelectKBest(f_classif,k=70).fit_transform(X_test,y_test)
print('X_train:', X_train.shape, X_train.dtype)
print('y_train:', y_train.shape, y_train.dtype)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)


# ## Using the MGCForest
# 
# Creates a simple *MGCForest* with 2 random forests for the *Multi-Grained-Scanning* process and 2 other random forests for the *Cascade* process.

# In[4]:


mgc_forest = MGCForest(
    estimators_config={
        'mgs': [{
            'estimator_class': ExtraTreesClassifier,
            'estimator_params': {
                'n_estimators': 100,
                'min_samples_split': 21,
                'n_jobs': -1,
            }
        }, {
            'estimator_class': RandomForestClassifier,
            'estimator_params': {
                'n_estimators': 100,
                'min_samples_split': 21,
                'n_jobs': -1,
            }
        }],
        'cascade': [{
            'estimator_class': ExtraTreesClassifier,
            'estimator_params': {
                'n_estimators': 1000,
                'min_samples_split': 11,
                'max_features': 1,
                'n_jobs': -1,
            }
        }, {
            'estimator_class': ExtraTreesClassifier,
            'estimator_params': {
                'n_estimators': 1000,
                'min_samples_split': 11,
                'max_features': 'sqrt',
                'n_jobs': -1,
            }
        }, {
            'estimator_class': RandomForestClassifier,
            'estimator_params': {
                'n_estimators': 1000,
                'min_samples_split': 11,
                'max_features': 1,
                'n_jobs': -1,
            }
        }, {
            'estimator_class': RandomForestClassifier,
            'estimator_params': {
                'n_estimators': 1000,
                'min_samples_split': 11,
                'max_features': 'sqrt',
                'n_jobs': -1,
            }
        }]
    },
    stride_ratios=[1.0 / 4, 1.0 / 9, 1.0 / 16],
)

mgc_forest.fit(X_train, y_train)
joblib.dump(mgc_forest, 'mgc_model.sav')
mgc_forest = joblib.load('mgc_model.sav')
# In[5]:


y_pred = mgc_forest.predict(X_test)

print('Prediction shape:', y_pred.shape)
print(
    'Accuracy:', accuracy_score(y_test, y_pred),
    'F1 score:', f1_score(y_test, y_pred, average='weighted')
)


