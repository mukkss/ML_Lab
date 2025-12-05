!pip install MiniSom
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from minisom import MiniSom


iris = load_iris()
data = iris.data
feature_names = iris.feature_names


scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)


df = pd.DataFrame(data, columns=feature_names)
print(df.head())


som_x, som_y = 10, 10
som = MiniSom(som_x, som_y, data.shape[1], sigma=1.0, learning_rate=0.5)


som.random_weights_init(scaled_data)
som.train_random(scaled_data, 1000)


plt.figure(figsize=(10, 10))
plt.pcolor(som.distance_map().T, cmap='bone_r')
plt.colorbar()
plt.title('SOM U-Matrix Distance Map')
plt.show()
