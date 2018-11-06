#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics



X= np.array([[1,1],[1.1,1.1],[1.2,1.2],[1.3,1.3],[1.4,1.4],[1.5,1.5],[1.6,1.6],
   [2,2], [2.1,2.1], [2.2,2.2], [2.3,2.3], [2.4,2.4], [2.5,2.5], [2.6,2.6]])
y=[1,1,1,1,1,1,1,
   0,0,0,0,0,0,0]


kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print("集群中心的坐標:",kmeans.cluster_centers_)


print("預測:",kmeans.predict(X))
print("實際:",y)

print("預測[1, 1],[2.3,2.1]:",kmeans.predict([[1, 1],[2.3,2.1]]))





plt.plot(X[:7,0], X[:7,1], 'yx' )
plt.plot(X[7:,0], X[7:,1], 'g.' )


plt.plot(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 'ro')


plt.xticks(())
plt.yticks(())
plt.show()


