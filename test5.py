# %%
#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#import the dataset
df = pd.read_csv('/home/chenhsi/Projects/Woven planet/iris.data')
df.head(10)


#select all four features (sepal length, sepal width, petal length, and petal width) 
x = df.iloc[:, [0,1,2,3]].values

# kmeans5 = KMeans(n_clusters=5)
# y_kmeans5 = kmeans5.fit_predict(x)
# print(y_kmeans5)
# print(kmeans5.cluster_centers_)

Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()


kmeans3 = KMeans(n_clusters=3)
y_kmeans3 = kmeans3.fit_predict(x)
print(y_kmeans3)
plt.scatter(x[:,0],x[:,1],c=y_kmeans3,cmap='rainbow')
plt.show()