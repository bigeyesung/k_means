import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
df = pd.read_csv('/home/chenhsi/Projects/Woven planet/iris_train.csv')
x = df.iloc[:, [0,1,2,3]].values
y = df.iloc[:, [-1]].values
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.15, random_state=42)
kmeans3 = KMeans(n_clusters=3)
y_kmeans3 = kmeans3.fit_predict(X_train)
print(y_kmeans3)
print(kmeans3.cluster_centers_)