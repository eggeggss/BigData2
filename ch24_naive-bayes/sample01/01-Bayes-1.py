#Import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
import numpy as np

#assigning predictor and target variables
x=np.array([[-3,7],[-1,5], [-1,2], [-2,0], [-2,3], [-4,0], [1,1], [1,1], [2,2], [2,7], [4,1], [2,7]])
y=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
#Create a Gaussian Classifier
model = GaussianNB()
# Train the model using the training sets 
model.fit(x,y)
#Predict Output
predictValue=np.array([[1,2],[3,4]])
predicted= model.predict(predictValue)
print(predicted)



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
