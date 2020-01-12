import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))

plt.title('Dendogram method')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5,affinity = 'euclidean',linkage='ward')

y_hc = hc.fit_predict(X)
print(y_hc)

plt.scatter(X[y_hc==0,0],X[y_hc==0,1], s =100, c='red', label='careful')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1], s =100, c='blue', label='standard')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1], s =100, c='green', label='target')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1], s =100, c='cyan', label='careless')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1], s =100, c='magenta', label='sensible')
#plt.scatter(X[y_hc==5,0],X[y_hc==5,1], s =100, c='yellow', label='New')

plt.title('Cluster of clients')
plt.xlabel('Annual Income k$')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()