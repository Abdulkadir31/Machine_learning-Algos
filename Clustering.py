import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans

x = np.array([[1,2],
             [1.5,1.8],
             [5,8],
             [8,8],
             [1,0.6],
             [9,11]])

clf = KMeans(n_clusters=2)
clf.fit(x)
centroid = clf.cluster_centers_
labels = clf.labels_
colors = ["g.","r.","c.","b.","k.","o."]

print(labels)
print(centroid)

for i in range(len(x)):
    plt.plot(x[i][0],x[i][1],colors[labels[i]],markersize = 10)
plt.scatter(centroid[:,0],centroid[:,1],marker = 'x',s = 20, linewidths=5,color = 'k')
plt.show()

# plt.scatter(x[:,0],x[:,1],s=10,linewidths=5)
# plt.show()

