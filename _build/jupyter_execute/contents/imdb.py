#!/usr/bin/env python
# coding: utf-8

# # IMDB 評論分類  
# 評論分成 positive 或是 negative 兩類，是一個二元分類問題。

# In[1]:


import tensorflow as tf
from tensorflow import keras


# In[2]:


imdb = keras.datasets.imdb

# 參數 num_words=10000 保留資料裡面出現最高頻率的 10000 個詞，低頻率出現的被移除
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# In[3]:


print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# show the first data
print(train_data[0])
print(len(train_data[0]))


# ## 探勘數據  
# 從上述展示可以看到我們訓練集有 25000 筆資料，label 有 25000 種  
# 而第一筆電影評論長度為 218，且都被轉換成數字  
# 然而每則電影評論的長度不盡相同，如下：

# In[4]:


print(len(max(train_data, key=len))) # 印出最長訓練評論的 size
print(len(max(test_data, key=len))) # 印出最長測試資料的 size
len(train_data[0]), len(train_data[1]), len(train_data[2])


# 而評論的標籤皆為 0 或是 1 的整數值，0 代表負面評論，1 代表正面評論  
# 如下我們展示前 20 筆標籤：

# In[5]:


train_labels[0:20]


# ## 準備數據  
# 這裡我們需要先將影評內容的 numpy array 轉換成 tensor，官方文件示範的方法為填充數據來讓每一則評論的長度相同，因為輸入到神經網路的 tensor 必須是相同的長度。這裡使用的方法為用 [pad sequences 函數](https://tensorflow.google.cn/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences)

# In[6]:


train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       padding='post',
                                                       maxlen=256)
# 經過處理後長度會相同
len(train_data[0]), len(train_data[1]), len(train_data[2])
                                            


# In[7]:


print(train_data[0]) # 長度不夠的補 0 補到 256


# ## 建構模型  
# 須考慮你需要多少層的模型，模型每層裡面要有多少 hidden units

# In[8]:


# 輸入的詞彙數量為 10000 種最高頻率出現的詞
vocab_size = 10000

model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.summary()


# In[9]:


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# ## 創建驗證集  
# 先從訓練集分出一些資料來做驗證 validating  
# 測試集要拿來做測試，因為要拿模型沒看過的資料來做測試 testing
# 
# 

# In[10]:


x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


# ## 訓練模型
# 

# In[11]:


history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=40,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=0
)


# ## 評估模型  
# 計算出損失值 loss，以及準確率 accuracy

# In[12]:


results = model.evaluate(test_data, test_labels, verbose=2)
print(results)


# ## 創建一個準確率和損失值隨時間變化的圖表  
# `model.fit()` 有一個 `history` 的屬性

# In[13]:


history_dict = history.history
history_dict.keys()


# In[14]:


import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# “bo”代表 "藍點"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b代表“藍色實線”
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[15]:


plt.clf()   # 清除數字

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# 從上述兩個圖看出大概在 epochs=15 的時候，即可訓練完成，再多可能會進入一個 over-fitting 的狀態。不過官方文件是說 epochs=20。

# 參考資料  
# [官方文件](https://www.tensorflow.org/tutorials/keras/text_classification)
