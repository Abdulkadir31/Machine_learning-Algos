import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
import warnings
from collections import Counter
import pandas as pd
import random
style.use('fivethirtyeight')


def k_nearest_neighbours(data, predict, k = 3):
    if len(data) >= k:
        warnings.warn('Value of K is less than total groups')

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    #most_common(1) -> NO OF ELEMENTS, [0] -> TUPLE POSITION, [0] -> LIST (0 for group and 1 count of neighbours)
    confidence = Counter(votes).most_common(1)[0][1]/k
    return vote_result,confidence


df = pd.read_csv('cancer.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[ :-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)): ]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote,confidence = k_nearest_neighbours(train_set,data,k=5)
        if vote == group:
            correct += 1
        else:
            print(confidence)
        total += 1
print("Accuracy :",correct/total)


### Plotting the Graph
# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0],ii[1],s=100,color=i)
# plt.scatter(new_features[0],new_features[1],s=100,color=result)
# plt.show()