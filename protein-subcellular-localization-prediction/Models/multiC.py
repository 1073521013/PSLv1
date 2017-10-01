
# coding: utf-8

# ## Multiclass Classification

# According to Wikipedia:
# > In machine learning, **multiclass** or **multinomial classification** is the problem of classifying instances into one of the more than two classes (classifying instances into one of the two classes is called binary classification).

# For instance, you are given an Apple product and you are told to classify it into one of the following:
# 
# 1. Apple Watch
# 2. Apple TV
# 3. iMac
# 4. iPad
# 5. iPhone
# 6. iPod
# 7. MacBook
# 
# So, this is a type of multiclass classification problem. 
# 
# In this post, let's learn how to approach a multiclass classification problem. We will be using a dataset that has 5 categories to which we should classify the given data. 

# Let's import all the necessary libraries. 

# In[183]:


# imports

# numpy, lambda, matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#get_ipython().magic(u'matplotlib inline')

# scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cross_validation import ShuffleSplit
from sklearn.learning_curve import learning_curve
from sklearn.manifold.t_sne import TSNE

# I will be explaining the purpose of each library as we approach the problem.


# Let's read the input data

# In[154]:


# train and test data
# here we will be using pandas library to read the input data in the form of CSV

train = pd.read_csv('../datasets/dataset_ilov/train.csv')
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

y_train = train['class']
names_train = train['name']
X_train = train.drop(['class', 'name', 'sequence'], axis=1)

y_test = test['class']
names_test = test['name']
X_test = test.drop(['class', 'name', 'sequence'], axis=1)
# Let's see how the training set looks after some trimming.

# In[36]:


# count of values under each category

label_count = y_train.value_counts().sort_index(ascending=True)
print label_count


# In[55]:


# visualizing the training labels

label_count.plot(kind='bar', color='skyblue')
plt.xlabel("categories")
plt.ylabel("Count of data")
plt.title("Count of data in each category")
plt.grid()

# StandardScaler brings down the range of values a single digit, where the new values are **0**. 
# StandardScaler applies a normal distribution, with **mean = 0** and **standard deviation = 1**.

# In[89]:


# standardize the data
std_scaler = StandardScaler().fit(X_train)

# transform train and test set using standardization
X_train_std = std_scaler.transform(X_train)
X_test_std = std_scaler.transform(X_test)


# Let's visualize the range of datapoints in training set

# In[94]:


# range of values in training dataset after standardization

plt.hist(X_train_std, 30, alpha=1)
plt.xlabel('Range of the data points')
plt.grid(True)
plt.show()


# Standardization worked!
# 
# The datapoints are now distributed between **-4** and **4**. 
# 
# Next, let's convert the **y_train** and **y_test** pandas series to numpy arrays.

# In[100]:


# converting from pandas series to numpy array

y_train = np.asarray(y_train, dtype='float64')
y_test = np.asarray(y_test, dtype='float64')


# So, all the major cleaning process is done, next we need to choose a classification algorithm to train it. 
# 
# But, before that we need to know what are the appropriate hyper paramters that our learning algorithm **KNeighborsClassifier** works efficiently.
# 
# In-order to choose the appropriate hyperparameters, let's using **GridSearchCV** from **model_selection**.

# In[107]:


# an instane of KNNeighborsClassifier
estimator = KNeighborsClassifier()

# cross-validation using ShuffleSplit
# we do this so the GridSearchCV will train and test the dataset to find the appropriate hyperparameters
cv = ShuffleSplit(int(len(X_train_std)), n_iter=10, random_state=0, test_size=0.2)

# instance of GridSearchCV
classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=dict(n_neighbors=[5,6,7,8,9,10,11,12,13,14,15]))

# let's fit the classifier with the training set
classifier.fit(X_train_std, y_train)


# That took a while to figure out the appropriate **n_neighbors** value 
# 
# Let's check the value of **n_neighbors**

# In[108]:


print classifier.best_params_


# Looks like assigning **n_neighbors** to **15** is a good choice.
# 
# Let's train the KNeighborsClassifier with the best params value

# In[112]:


# instance of KNeighborsClassifier

#estimator = KNeighborsClassifier(n_neighbors=6)
estimator = RandomForestClassifier(n_estimators=500, criterion='entropy', max_features=8, max_depth=25,
                                    min_samples_leaf = 1, bootstrap=True, oob_score=True, n_jobs=30, random_state=0)
#estimator = GradientBoostingClassifier(learning_rate=0.05,max_depth=15,min_samples_leaf=2,min_samples_split=0.5,n_estimators=500)

# In[113]:


# fit the training data

estimator.fit(X_train_std, y_train)


# In[114]:


# testing on validation set

y_pred = estimator.predict(X_test_std)


# Now, let's print **accuracy_score**, **confusion_matrix** and **classification_report** from **sklearn.metrics**

# In[119]:


# printing out some metrics

print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
print("-"*50)
print("Confusion Matrix: \n{}".format(confusion_matrix(y_test, y_pred)))
print("-"*50)
print("Classification Report: \n{}".format(classification_report(y_test, y_pred)))


# All the above scores are fine, let's have a look at the learning curve.

# In[141]:


def leaning_curve_plot(estimator, X, y, cv):
    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Traning Example")
    plt.ylabel("Score")
    plt.grid()
    
    # compute mean and standard deviation
    train_sizes, train_scores, test_scores = learning_curve( estimator, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # fill_between 
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')
    
    # plotting 
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label="Trainig score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label="Cross Validation score")

    plt.legend(loc="best")
    
    return plt


# In[142]:


plt = leaning_curve_plot(estimator, X_train_std, y_train, cv)
plt.show()


# Looks like the estimator's score goes high when the dataset size increases.

# In[150]:
# Time to classify the blind set
blind = pd.read_csv('../datasets/dataset_csv/blind.csv')
blind_x = blind.drop(['name', 'sequence'], axis=1)
X_test = std_scaler.transform(blind_x)

y_pred = estimator.predict(X_test)
print y_pred
# saving the output to a csv
#output = pd.DataFrame({'id': test_ids, 'label': y_pred})
## printing output
#output.head(4)

# Let's look at the decision boundary for the given dataset

# In[185]:


# reducing the dimensionality from 15 to 2
X_train_embedded = TSNE(n_components=2).fit_transform(X_train_std)
print X_train_embedded.shape
model = GradientBoostingClassifier(learning_rate=0.05,max_depth=15,min_samples_leaf=2,
                                   min_samples_split=0.5,n_estimators=500).fit(X_train_std, y_train)
y_predicted = model.predict(X_train_std)

# create meshgrid
resolution = 150 # 150x150 background pixels
X2d_xmin, X2d_xmax = np.min(X_train_embedded[:,0]), np.max(X_train_embedded[:,0])
X2d_ymin, X2d_ymax = np.min(X_train_embedded[:,1]), np.max(X_train_embedded[:,1])
xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))

# approximate Voronoi tesselation on resolution x resolution grid using 1-NN
background_model = GradientBoostingClassifier(learning_rate=0.05,max_depth=15,
                                              min_samples_leaf=2,min_samples_split=0.5,n_estimators=500).fit(X_train_embedded, y_predicted) 
voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
voronoiBackground = voronoiBackground.reshape((resolution, resolution))

#plot
plt.contourf(xx, yy, voronoiBackground)
plt.scatter(X_train_embedded[:,0], X_train_embedded[:,1], c=y_train)
plt.show()


# Here, we see a clear classification of the datapoints into different categories. 
