# -*- coding: utf-8 -*-
"""Fraud-Detection-using-SOM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wbdgImNy1TxbuKeYxeMTU03c70n63UKp
"""

!pip install MiniSom

#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show

df= pd.read_csv("Credit_Card_Applications.csv")
df.head()

X = df.iloc[:,:14]
y=df["Class"]

# Like all Unsupervised algorithms, in SOM also we should scale our features for more accurate results
sc = MinMaxScaler(feature_range = (0,1))
sc.fit(X)
X=sc.transform(X)
type(X)
# minisom expects input in numpy array, so we will not convert X to pandas dataframe
som = MiniSom(x=10, y=10, input_len= 14, sigma= 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

bone()
pcolor(som.distance_map().T)
# distance map as backgroundcolorbar()
markers = ['o','s']
colors = ['r','g']
for i, x in enumerate(X):
  w = som.winner(x)
  # w[0], w[1] will place the marker at bottom left corner of the rectangle. 
  #Let us add 0.5 to both of these to plot the market at the center of the rectange.    
  plot(w[0] + 0.5, 
       w[1] + 0.5,
       #Target value 0 will have marker "o" with color "r"
       #Target value 1 will have marker "s" with color "g"         
       markers[y[i]],         
       markeredgecolor = colors[y[i]],         
       markerfacecolor = 'None',
       #No color fill inside markers         
       markersize = 10,         
       markeredgewidth = 2)
show()

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(3,2)]), axis = 0)

frauds_orig = sc.inverse_transform(frauds)
frauds_orig= frauds_orig[:,0].astype("int64")   
frauds_orig