import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris


iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)


k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)


X['cluster'] = kmeans.labels_
print("cluster centers:\n", kmeans.cluster_centers_)


plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=X['cluster'], cmap='viridis', s=40, alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=50, label='centroids')


plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("K-Means clustering (Iris Dataset)")
plt.legend()
plt.show()
