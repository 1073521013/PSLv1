# coding: utf-8
# In[1]:

import pandas as pd
import numpy as np
from GCForest import gcForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
# In[2]:
# loading the data

train = pd.read_csv('../datasets/dataset_iloc/train.csv')
test = pd.read_csv('../datasets/dataset_iloc/test.csv')
train.loc[train["class"]=="chloroplast","class"]=0
train.loc[train["class"]=="cytoplasmic","class"]=1
train.loc[train["class"]=="ER","class"]=2
train.loc[train["class"]=="extracellular","class"]=3
train.loc[train["class"]=="Golgi","class"]=4
train.loc[train["class"]=="lysosomal","class"]=5
train.loc[train["class"]=="mitochondrial","class"]=6
train.loc[train["class"]=="nuclear","class"]=7
train.loc[train["class"]=="peroxisomal","class"]=8
train.loc[train["class"]=="plasma_membrane","class"]=9
train.loc[train["class"]=="vacuolar","class"]=10

test.loc[test["class"]=="chloroplast","class"]=0
test.loc[test["class"]=="cytoplasmic","class"]=1
test.loc[test["class"]=="ER","class"]=2
test.loc[test["class"]=="extracellular","class"]=3
test.loc[test["class"]=="Golgi","class"]=4
test.loc[test["class"]=="lysosomal","class"]=5
test.loc[test["class"]=="mitochondrial","class"]=6
test.loc[test["class"]=="nuclear","class"]=7
test.loc[test["class"]=="peroxisomal","class"]=8
test.loc[test["class"]=="plasma_membrane","class"]=9
test.loc[test["class"]=="vacuolar","class"]=10
df = pd.DataFrame(train["class"])
df.to_csv('../datasets/dataset_iloc/train_int.csv', header=False, index=False)
df = pd.DataFrame(test["class"])
df.to_csv('../datasets/dataset_iloc/test_int.csv', header=False, index=False)

# Split into labels, names and data
y_tr = train['class']
names_train = train['name']
X_tr = train.drop(['class', 'name', 'sequence'], axis=1)

y_te = test['class']
names_test = test['name']
X_te = test.drop(['class', 'name', 'sequence'], axis=1)

# In[3]:


gcf = gcForest(n_cascadeRFtree=1000,n_mgsRFtree=1000,shape_1X=72,
               window=[5,9,18],min_samples_mgs=10, min_samples_cascade=7)
joblib.dump(gcf, 'gcf_model.sav')
#X_tr = X_tr.as_matrix()
#X_te = X_te.as_matrix()
gcf = joblib.load('gcf_model.sav')
std_scaler = StandardScaler().fit(X_tr)
# transform train and test set using standardization
X_tr = std_scaler.transform(X_tr)
X_te = std_scaler.transform(X_te)
y_tr = np.asarray(y_tr, dtype='int')
y_te = np.asarray(y_te, dtype='int')
gcf.fit(X_tr, y_tr)


pred_X = gcf.predict(X_te)

# In[5]:


# evaluating accuracy
accuracy = accuracy_score(y_true=y_te, y_pred=pred_X)
print('gcForest accuracy : {}'.format(accuracy))
# In[10]:

joblib.dump(gcf, 'gcf_model.sav')

# In[11]:
gcf = joblib.load('gcf_model.sav')


# In[14]:
#gcf = gcForest(n_cascadeRFtree=300,n_mgsRFtree=200,shape_1X=72,
#               window=[5,9,18],min_samples_mgs=10, min_samples_cascade=7)
#gcf = gcForest(n_cascadeRFtree=500,n_mgsRFtree=500,shape_1X=72,window=[5,9,18],
#               min_samples_mgs=20, min_samples_cascade=10)
X_tr_mgs = gcf.mg_scanning(X_tr, y_tr)

# In[15]:
X_te_mgs = gcf.mg_scanning(X_te)

# In[18]:
#gcf = gcForest(n_cascadeRFtree=500,n_mgsRFtree=500,shape_1X=72,window=[5,9,18],
#               min_samples_mgs=20, min_samples_cascade=10)
_ = gcf.cascade_forest(X_tr_mgs, y_tr)

# In[19]:
pred_proba = gcf.cascade_forest(X_te_mgs)
tmp = np.mean(pred_proba, axis=0)
preds = np.argmax(tmp, axis=1)
accuracy_score(y_true=y_te, y_pred=preds)

# In[20]:
#<h3>Skipping mg_scanning</h3>
gcf = gcForest(tolerance=0.0, min_samples_cascade=20)
_ = gcf.cascade_forest(X_tr, y_tr)

# In[21]:
pred_proba = gcf.cascade_forest(X_te)
tmp = np.mean(pred_proba, axis=0)
preds = np.argmax(tmp, axis=1)
accuracy_score(y_true=y_te, y_pred=preds)

