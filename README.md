# Fraud-Detection-using-SOM

A Self-Organizing Map (SOM) is a type of artificial neural network that is used for unsupervised learning. It maps input data points to a low-dimensional representation, such as a grid of neurons, in a way that preserves the topological structure of the data. This allows it to detect patterns and cluster similar data points together. SOMs are commonly used for dimensionality reduction, visualization, and clustering tasks in various fields such as computer vision, natural language processing, and finance. SOMs are useful for data exploration, clustering, and pattern recognition. They can be used to identify patterns in large datasets, such as customer behavior or stock prices. They are also useful for data visualization, as they can be used to represent data in a low-dimensional space, making it easier to understand the underlying structure of the data.

### Importing Libraries

```javascript I'm A tab
!pip install MiniSom
```
```javascript I'm A tab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show
```

### Load Data
```javascript I'm A tab
df= pd.read_csv("Credit_Card_Applications.csv")
df.head()
```

```javascript I'm A tab
X = df.iloc[:,:14]
y=df["Class"]
```

### Scaling features
```javascript I'm A tab
sc = MinMaxScaler(feature_range = (0,1))
sc.fit(X)
X=sc.transform(X)
type(X)
```

### Create the model
```javascript I'm A tab
som = MiniSom(x=10, y=10, input_len= 14, sigma= 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

bone()
pcolor(som.distance_map().T)

markers = ['o','s']
colors = ['r','g']
for i, x in enumerate(X):
  w = som.winner(x)
  plot(w[0] + 0.5, 
       w[1] + 0.5,
       markers[y[i]],         
       markeredgecolor = colors[y[i]],         
       markerfacecolor = 'None',
       markersize = 10,         
       markeredgewidth = 2)
show()
```
![image](https://user-images.githubusercontent.com/82306595/216673551-deaf9021-d76c-46db-a0b2-aa9b18f125c8.png)


### Finding the Frauds

```javascript I'm A tab
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(3,2)]), axis = 0)
```

### CustomerIDs of Fraudulent customers
```javascript I'm A tab
frauds_orig = sc.inverse_transform(frauds)
frauds_orig= frauds_orig[:,0].astype("int64")   
frauds_orig
```

![image](https://user-images.githubusercontent.com/82306595/216673766-fb814880-2735-4ad9-a054-292f7c59098a.png)
