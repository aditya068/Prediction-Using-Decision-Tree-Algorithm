#!/usr/bin/env python
# coding: utf-8

# In[7]:


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree


# In[19]:


import pandas as pd
from sklearn.datasets import load_iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
X_train, X_test, Y_train, Y_test = train_test_split(df[data.feature_names], df['target'], random_state=0)
print(df.head(5))
y=iris.target
print(y)


# In[20]:


# Step 1: Import the model you want to use
# This was already imported earlier in the notebook so commenting out
#from sklearn.tree import DecisionTreeClassifier
# Step 2: Make an instance of the Model
clf = DecisionTreeClassifier()
clf.fit(df,y)
# Step 3: Train the model on the data
clf.fit(X_train, Y_train)
print('Decision Tree Classifier Created')
# Step 4: Predict labels of unseen (test) data
# Not doing this step in the tutorial
# clf.predict(X_test)


# In[12]:


tree.plot_tree(clf);


# In[37]:


from six import StringIO
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(clf,
               feature_names = iris.feature_names, 
               class_names=cn,
               filled = True, rounded=True);
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
fig.savefig('imagename.png')


# In[ ]:




