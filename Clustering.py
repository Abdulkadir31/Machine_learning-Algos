import numpy as np
import pandas as pd
from sklearn import preprocessing,neighbors
from sklearn.model_selection import train_test_split

df = pd.read_csv('cancer.data.txt')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train,X_test,y_train,y_test =train_test_split(X, y, test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
print(accuracy)

predict = np.array([[8,2,7,7,7,6,3,7,7],[4,2,1,1,1,4,3,1,1]])
predict = predict.reshape(len(predict),-1)
prediction = clf.predict(predict)
print(prediction)