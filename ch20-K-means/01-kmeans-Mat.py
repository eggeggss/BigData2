#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

X= np.array([[1,1],[1.1,1.1],[1.2,1.2],[1.3,1.3],[1.4,1.4],[1.5,1.5],[1.6,1.6],
   [2,2], [2.1,2.1], [2.2,2.2], [2.3,2.3], [2.4,2.4], [2.5,2.5], [2.6,2.6]])
y=[1,1,1,1,1,1,1,
   0,0,0,0,0,0,0]



plt.plot(X[:7,0], X[:7,1], 'yx' )
plt.plot(X[7:,0], X[7:,1], 'g.' )


plt.ylabel('H cm')
plt.xlabel('W cm')
plt.legend(('A','B'),
           loc='upper right')
plt.show()