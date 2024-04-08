import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv("/content/sample_data/partywise.csv")
plt.scatter(data["Party"], data["Won"])
plt.xlim(-180,180)
x = data.iloc[:,1:3]
data.dtypes
kmeans = KMeans(3)
kmeans.fit(x)
identified_clusters = kmeans.fit_predict(x)
identified_clusters
data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters
plt.scatter(data_with_clusters['Won'],data_with_clusters['Party'],c=data_with_clusters['Won'],cmap='rainbow')