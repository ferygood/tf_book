#!/usr/bin/env python
# coding: utf-8

# # 單顆神經元  
# 這裡示範最基本的 TensorFlow 架構，由一顆神經元組成的網路  
# 用簡單的 keras.Sequential model 輸入一單位的 input, input shape 也是 1  
# sgd 優化為 stochastic gradient descent  
# train 500 個 epochs, 答案非 19 而是近似 19 原因是模型會加入誤差以防 over-fitting

# In[1]:


from tensorflow import keras
import numpy as np


# In[2]:


model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)
model.predict([10.0])

