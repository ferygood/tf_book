#!/usr/bin/env python
# coding: utf-8

# # Fashion MNIST 基礎分類  
# Fashion MNIST 有 10 個種類的圖，總共 70k 的圖像，圖像的尺寸為 28x2，每個分類大約有 7k 張，這樣就足夠訓練一個神經網路了  
# 用數字 09 代表短靴，這樣可以避免誤差和容易電腦計算  
# * 本篇教學適用於你想要做小 size 的影像識別神經網路
# * 缺陷是圖片都是 28 x 28 灰階，且圖片都在正中間，較不符合實際應用

# In[1]:


import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[2]:


# 瀏覽數據
print(train_images.shape)
print(len(train_labels))
print(test_images.shape)
print(len(test_labels))


# 概述模型層  
# 1. 28x28 先拉成線性的輸入  
# 2. 使用 relu 來分析 128 筆函數的輸出結果，是不是每個函數都能夠輸出正確的答案，以這個例子應該是要 09  
# 3. 用 softmax 選取最高的類別，我們有 10 個類別，所以如果圖片是短靴，理論上09的機率要是最高，所以 softmax 函數下，09應該最接近 1

# In[3]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


# In[4]:


model.fit(train_images, train_labels, epochs=10, verbose=0)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(test_loss)
print(test_acc)


# In[5]:


# 如果想要測試
# predictions = model.predict(my_images)


# 參考資料  
# [TensorFlow YouTube 影片](https://www.youtube.com/watch?v=ifj5bAzrzMw&list=PLQY2H8rRoyvwr-3IlvJXA1JyOlpcbIGa1&index=2&ab_channel=TensorFlow)  
# [官方文件](https://www.tensorflow.org/tutorials/keras/classification)
# 
