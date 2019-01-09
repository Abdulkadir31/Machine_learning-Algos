from math import sqrt
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
import warnings
from collections import Counter
style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0],ii[1],s=100,color=i)
# plt.scatter(new_features[0],new_features[1],s=100)
# plt.show()

# Euclidean_distance = sqrt( (plot1[0] - plot2[0] )**2 + (plot1[1] - plot2[1] )**2 )
# print(Euclidean_distance)