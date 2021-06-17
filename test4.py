# %%
import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


raw_data = make_blobs(n_samples = 200, n_features = 2, centers = 4, cluster_std = 1.8)

#Visualize the data
plt.scatter(raw_data[0][:,0], raw_data[0][:,1])
plt.scatter(raw_data[0][:,0], raw_data[0][:,1], c=raw_data[1])
plt.legend()
plt.show()
#Build and train the model
from sklearn.cluster import KMeans
model = KMeans(n_clusters=4)
model.fit(raw_data[0])

print(model.labels_)
print(model.cluster_centers_)

#PLot the predictions against the original data set
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('Our Model')
ax1.scatter(raw_data[0][:,0], raw_data[0][:,1],c=model.labels_)
ax2.set_title('Original Data')
ax2.scatter(raw_data[0][:,0], raw_data[0][:,1],c=raw_data[1])
plt.legend()
plt.show()