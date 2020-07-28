#!/usr/bin/env python
# coding: utf-8

# # IRIS DATASET

# In[107]:


import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.offline as pyo
import cufflinks as cf
from plotly.offline import init_notebook_mode,plot,iplot

import pickle
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pyo.init_notebook_mode(connected=True)
cf.go_offline()


# In[3]:


#Loading the data 


# In[4]:


iris = pd.read_csv(r'S:\GN\Assigned1_Dim_Red_Clustering_750\IRIS.csv')


# In[5]:


iris


# In[6]:


#id is not a useful feature so we drop it
#alreay dropped in the dataset
#iris.drop('Id' , axis =1)


# In[ ]:





# In[ ]:





# # Visualizing our data

# 1. SCATTER PLOT

# In[7]:


px.scatter(iris , x = "species" , y = "sepal_length" , size = "sepal_length", color = "sepal_length")


# In[8]:


px.scatter(iris , x = "species" , y = "petal_length" , size = "petal_length",color = "petal_length")


# In[9]:


px.scatter(iris , x = "species" , y = "sepal_width" , size = "sepal_width",color = "sepal_width")


# In[10]:


px.scatter(iris , x = "species" , y = "petal_width" , size = "petal_width" ,color = "petal_width")


# In[11]:


px.scatter(iris , x = "species" , y = "sepal_length" ,size = "sepal_width" ,color = "petal_length")


# In[12]:


px.scatter(iris , x = "species" , y = "sepal_length" , size = "petal_width" ,  color ="petal_length")


# In[13]:


px.scatter(iris , x = "species" , y = "sepal_length" , size = "sepal_width",color="sepal_length")


# In[ ]:





# ## 2. Bar plot

# seeing the difference between matplotlib ploting and plotly for the barplot

# In[14]:


#using matplotlib


# In[15]:


plt.bar(iris["species"],iris["petal_width"])
#we cannot hover over the data


# In[16]:


#using plotly


# In[17]:


px.bar(iris , x = "species" , y="petal_length")
#here we can hover the data and we get can differentaite the boxplot of each
#value if petal_legth


# In[18]:


px.bar(iris , x = "species" , y="petal_width")


# In[19]:


px.bar(iris , x = "species" , y="sepal_length")


# In[20]:


px.bar(iris , x = "species" , y="sepal_width")


# In[ ]:





# ## 3.Line Graph

# In[21]:


px.line(iris , x = "species" , y = "petal_length")
#we see we have for setosw - it is a straight upward line in the range 1-1.9
#for versicolor 3-5.1
#for verginica 4.5 - 6.9


# In[22]:


px.line(iris , x = "species" , y = "sepal_length")
#for sepal length the range for diff species of the iris flower is 
#setosa - 4.3-5.8
#versicolor - 4.9-7
#verginica - 4.9 -7.5


# In[23]:


px.line(iris , x = "species" , y = "sepal_width")
#for sepal length the range for diff species of the iris flower is 
#setosa - 2.3-4.4
#versicolor - 2-3.4
#verginica - 2.2 - 3.8


# In[24]:


px.line(iris , x = "species" , y = "petal_width")
#for sepal length the range for diff species of the iris flower is 
#setosa - 0.1-0.6
#versicolor - 1-1.8
#verginica - 1.4 -2.5


# In[ ]:





# In[ ]:





# ## 4.Scatter Matrix

# In[25]:


px.scatter_matrix(iris , color = "species" , title = "IRIS" , dimensions = ["sepal_length" , "sepal_width" , "petal_length" , "petal_width"])
#sepal_width vs rest features
#sepal_length vs rest features
#petal_width vs rest features
#petal_length vs rest features


# In[ ]:





# In[ ]:





# # Training the model

# In[26]:


#putting all the numerical values in a dataframe X
X = iris.drop(["species"] , axis = 1)


# In[27]:


X


# In[28]:


#putting the species in another dataframe named y
y = iris['species']


# In[29]:


y


# In[30]:


#as the species data is categorical so we convert the categorical data into 
#numerical values , we use ONE HOT ENCODING


# In[31]:


#ONE-HOT LABEL ENCODING
from sklearn.preprocessing import LabelEncoder
label_e = LabelEncoder()
y = label_e.fit_transform(y)


# In[32]:


y


# In[33]:


#converting the dataframe into an array
X = np.array(X)


# In[34]:


X


# In[ ]:





# In[ ]:





# In[ ]:





# # Training And Testing

# Training the decision tree

# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


#spliiting the dataset into train and test
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 0)


# # 1.Decision Trees

# In[37]:


from sklearn import tree
dec_tree = tree.DecisionTreeClassifier()


# In[40]:


#training the decision tree model using the training data - 4 features and their corresponding speciesdec_tree.
dec_tree.fit(X_train , y_train)


# Testing , predicting

# In[41]:


#predicting our test set on the trained model
pred_DT = dec_tree.predict(X_test)


# In[42]:


#finding the accuracy of our model
from sklearn.metrics import accuracy_score
accuracy_DT = accuracy_score(y_test,pred_DT) *100


# In[43]:


accuracy_DT


# In[ ]:





# In[44]:


X_train.size
#(70/100)*150 = 105 , 105*4 (number of features of each record ) = 420


# In[45]:


y_test.shape


# In[46]:


X_train.shape


# In[47]:


X_test.size


# In[48]:


X_test.shape


# In[49]:


y_train.size


# In[50]:


y_train.shape


# In[51]:


y_test.size


# In[ ]:





# comparing the results we got from our model to the actual species

# In[52]:


pred_DT


# In[53]:


y_test


# In[54]:


np.array_equal(pred_DT, y_test)


# predicting the species using user i/p

# In[55]:


Category = ["Iris - Setosa" , "Iris - Versicolor" , "Iris - Virginica"]


# In[56]:


User_DT = np.array([[5.2,1.3,3.4,2.5]])


# In[57]:


U_pred = dec_tree.predict(User_DT)


# In[58]:


U_pred
#the output says 2 i.e Virginica


# In[59]:


#converting the array([2]) value to integer 
U_pred[0]


# In[60]:


print("The species for the input is :" , Category[int(U_pred[0])])


# # Visualizing our Decision Tree

# # 2. KNN Classifier (K-Nearest Neighbor)

# Scaling the features using standard scalar

# we dont use it in decision trees as we do not evaluate the distance , we just analyze the patterns and make the decisions

# In[61]:


from sklearn.neighbors import KNeighborsClassifier


# In[62]:


from sklearn.preprocessing import StandardScaler


# In[63]:


#we do not need to use standard scalar here because we have all the features having the same unit 
#but doing for the purpose of understanding
#Scaling down the test and training set features

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# In[64]:


X_train[0:4]


# In[65]:


X_train_std


# In[66]:


X_test[0:4]


# In[67]:


#comparing the first 4 enteries , we see the scaling ,now they all fall in a range
X_test_std


# In[68]:


knn = KNeighborsClassifier(n_neighbors = 5)
#our neighbors should be in a range 
#n_neighbors is the number of neighbors we want to use , we can loop through a range of values to know which can be the best 
#value for n


# In[69]:


knn.fit(X_train_std , y_train)


# In[70]:


#predicting on the test data

knn_predict = knn.predict(X_test_std)


# In[71]:


knn_predict


# In[72]:


#finding the accuracy of our model

knn_accuracy = accuracy_score(y_test,knn_predict)*100


# In[73]:


knn_accuracy


# to test on our input , we need to scale that down too , else it will just give us a single species everytime

# In[74]:


#trying with unscaled inputs
inp_knn = np.array([[5.5, 4.2, 1.4, 0.2]])


# In[75]:


pred_knn = knn.predict(inp_knn)


# In[76]:


Category[int(pred_knn)]

#but it actually is setosa , because we did not scale it down , that is why virginica


# In[77]:


#trying with unscaled inputs
inp_knn = np.array([[6.3 , 2.3 , 4.4 , 1.3]])
pred_knn = knn.predict(inp_knn)
Category[int(pred_knn)]

#it actually is versicolor , because we did not scale it down , that is why virginica 


# we see from the above 2 cases that no matter what the input is , it always gives virginica

# we now scale down the inputs too to predict the right specie

# In[78]:


inp_knn = np.array([[6.3 , 2.3 , 4.4 , 1.3]])
inp_knn_std = sc.transform(inp_knn)


# In[79]:


inp_knn_std


# In[80]:


pred_knn_std = knn.predict(inp_knn_std)


# In[81]:


pred_knn_std


# In[82]:


Category[int(pred_knn_std[0])]
#we now get the correct output


# In[ ]:





# In[83]:


inp_knn =  np.array([[5.5, 4.2, 1.4, 0.2]])


# In[84]:


inp_knn_std = sc.transform(inp_knn)


# In[85]:


inp_knn_std


# In[86]:


pred_knn_std = knn.predict(inp_knn_std)
print(pred_knn_std[0])
print(Category[int(pred_knn_std[0])])
#we now get the correct output


# In[ ]:





# In[87]:


k_range = range(1,26)
scores = {}
scores_list = []
for k in k_range :
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train_std , y_train)
    prediction_knn = knn.predict(X_test_std)
    #scores[k] = accuracy_score(y_test,prediction_knn)
    scores_list.append(accuracy_score(y_test,prediction_knn))


# In[88]:


#we get the accuracy score against each value of k
scores_list


# In[89]:


#we see it first increases from k=1 to k=3 then stays constant throughout k =3 to k=15 with the accuracy score of 97.77%
#but falls gradually from 15 to 25
plt.plot(k_range , scores_list)


# In[90]:


#we can train our model with n with diff values we can see the accuracy acc to the value of n


# In[91]:


px.line(k_range , scores_list)


# # 3. K-Means Algorithm (Unsupervised Learning)

# In[92]:


y


# In[93]:


colormap=np.array(['Red','green','blue'])
fig = plt.scatter(iris['petal_length'],iris['petal_width'],c=colormap[y],s=50)


# In[94]:


iris


# In[95]:


from sklearn.cluster import KMeans
km = KMeans(n_clusters=3,random_state=2,n_jobs=4)
km.fit(X)


# In[96]:


centers=km.cluster_centers_
print(centers)


# In[97]:


km.labels_


# In[98]:


Catagory_kmeans=['Iris-Versicolor', 'Iris-Setosa', 'Iris-Virginica']


# In[99]:


Catagory_kmeans


# In[103]:


colormap=np.array(['Red','green','blue'])
fig=plt.scatter(iris['petal_length'],iris['petal_width'],c=colormap[km.labels_],s=50)


# In[104]:


new_labels=km.labels_
fig,axes=plt.subplots(1,2,figsize=(16,8))
axes[0].scatter(X[:,2],X[:,3],c=y,cmap='gist_rainbow',edgecolor='k',s=150)
axes[1].scatter(X[:,2],X[:,3],c=y,cmap='jet',edgecolor='k',s=150)
axes[0].set_title('Actual',fontsize=18)
axes[1].set_title('Predicted',fontsize=18)


# In[105]:


#predicting on custom input value
X_km=np.array([[1 ,1, 1, 1]])
X_km_prediction=km.predict(X_km)
X_km_prediction[0]
print(Catagory_kmeans[int(X_km_prediction[0])])


# In[ ]:





# In[108]:


pickle.dump(dec_tree,open('dec_tree_model.pkl','wb'))
pickle.dump(knn,open('knn_model.pkl','wb'))
pickle.dump(km,open('kmeans_model.pkl','wb'))


# In[ ]:




