import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

x = np.array([[1,2],
             [1.5,1.8],
             [5,8],
             [8,8],
             [1,0.6],
             [9,11]])

colors = ["g","r","c","b","k","o"]

# plt.scatter(x[:,0],x[:,1],s=10,linewidths=5)
# plt.show()

class KMeans:
    def __init__(self,k=2, tol = 0.001, max_iter=300):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol

    def fit(self,data):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis = 0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]

                if np.sum((current_centroid - original_centroid)/original_centroid *100) > self.tol :
                    optimized = False

            if optimized:
                break


    def predict(self,data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

clf = KMeans()
clf.fit(x)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0],clf.centroids[centroid][1],marker = 'o', color = 'k', s = 10,linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1],marker = 'x',color = colors[classification],s = 50, linewidths=5)



unknowns  = np.array([[1,3],
                      [8,9],
                      [0,3],
                      [5,7]])
for unknown in unknowns:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0],unknown[1],marker = '*',color = colors[classification],s = 50,linewidths=5)

plt.show()