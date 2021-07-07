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
#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x=StandardScaler()  
x_train=st_x.fit_transform(x_train)    
x_test=st_x.transform(x_test)

#%%
#train the classifiert1=time.time()
t1=time.time()
from sklearn.neural_network import MLPClassifier
classifier=MLPClassifier(random_state=1, max_iter=500)
classifier.fit(x_train, y_train)
t2=time.time()
training_time=t2-t1

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
Accuracy=accuracy_score(y_test,Y_predict)