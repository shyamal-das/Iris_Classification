#import modules
import time 
import matplotlib.pyplot as plt  
import pandas as pd  
#importing dataset  
data_set= pd.read_csv('Iris.CSV')  
#Extracting Independent and dependent Variable  
x=data_set.iloc[:,0:4]
y=data_set.iloc[:,4:5]
#Splitting the dataset into training and testing set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size= 0.25, random_state=0)  

#%%
#Fitting Decision Tree classifier to the training set
t1=time.time()  
from sklearn import tree 
classifier= tree.DecisionTreeClassifier(criterion='gini', random_state=0)  
classifier.fit(x_train, y_train)
t2=time.time()
training_time=t2-t1

#%%
#visualize the tree
fig=plt.figure(figsize=(25,20))
_=tree.plot_tree(classifier,filled=True)
fig.savefig("decision_tree.png")

#%%
#Predicting the test set result
t3=time.time() 
Y_predict=classifier.predict(x_test)
t4=time.time()
prediction_time=t4-t3  

#%%
#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm=confusion_matrix(y_test, Y_predict)

#%%
#checking accuracy
from sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(y_test,Y_predict))

#%%
#visualize dataset
import seaborn

seaborn.lmplot('SL','SW', data=data_set, fit_reg=False, hue="Class", scatter_kws={"marker": "D", "s":50})
plt.title('SL vs SW')
plt.savefig('data_viz1.png', dpi=500)

seaborn.lmplot('PL','PW', data=data_set, fit_reg=False, hue="Class", scatter_kws={"marker": "D", "s":50})
plt.title('PL vs PW')
plt.savefig('data_viz2.png', dpi=500)

seaborn.lmplot('PL','SL', data=data_set, fit_reg=False, hue="Class", scatter_kws={"marker": "D", "s":50})
plt.title('PL vs SL')

seaborn.lmplot('PW','SW', data=data_set, fit_reg=False, hue="Class", scatter_kws={"marker": "D", "s":50})
plt.title('PW vs SW')

seaborn.lmplot('PL','SW', data=data_set, fit_reg=False, hue="Class", scatter_kws={"marker": "D", "s":50})
plt.title('PL vs SW')

seaborn.lmplot('PW','SL', data=data_set, fit_reg=False, hue="Class", scatter_kws={"marker": "D", "s":50})
plt.title('PW vs SL')