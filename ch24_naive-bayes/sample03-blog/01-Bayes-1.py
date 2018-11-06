#Import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
import numpy as np

#assigning predictor and target variables
x=np.array([[15,25,0,5,8,3],[35,40,1,3,3,2], [5,0,35,50,0,0], [1,5,32,15,0,0], [10,5,7,0,2,30], [5,5,5,15,8,32]])
y=np.array([0, 0, 1,1, 2,2])
#Create a Gaussian Classifier
model = GaussianNB()
# Train the model using the training sets 
model.fit(x,y)
#Predict Output
predictValue=np.array([[1,2,10,25,0,0]])
predicted= model.predict(predictValue)
print(predicted)

"""

import matplotlib.pyplot as plt


colormap = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'])

x1=x[:, 0]
print(x1)
y1=x[:, 1]
print(y1)

plt.scatter(x1, y1, s=50, c=colormap[y])

plt.plot(predictValue[:,0],predictValue[:,1],'rx')

plt.ylabel('Y')
plt.xlabel('X')
plt.show()
"""