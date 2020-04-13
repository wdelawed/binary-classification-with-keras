from keras.datasets import imdb 
from keras import  models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension = 10000) : 
    results = np.zeros((len(sequences),dimension)) 
    for i , sequence in enumerate(sequences) : 
        results[i,sequence]  = 1 
    return results 

x_train = vectorize_sequences(train_data) 
x_test  = vectorize_sequences(test_data) 

y_train = np.asarray(train_labels).astype('float32')
y_test  = np.asarray(test_labels).astype('float32')

model = models.Sequential() 
model.add(layers.Dense(16, activation = 'relu',input_shape=(10000,)))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1 , activation = 'sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

x_val = x_train[:10000] 
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:] 

history = model.fit(partial_x_train,partial_y_train,batch_size = 512,epochs=20,validation_data=(x_val,y_val)) 

history_dic = history.history 

val_loss = history_dic['val_loss'] 
loss = history_dic['loss'] 

epochs = range(1,len(loss)+1)

plt.plot(epochs,val_loss,'b',Label="Validation Loss")
plt.plot(epochs,loss,'bo',Label="Training Loss") 
plt.title("Losses Vs Epochs plot")
plt.xlabel("Epoch")
plt.ylabel("Loss") 
plt.legend()
plt.show()


val_acc = history_dic['val_acc'] 
acc = history_dic['acc'] 

plt.clf() 

plt.plot(epochs, val_acc, 'b', Label="Validation Accuracy")
plt.plot(epochs, acc, 'bo', Label="Training Accuracy") 
plt.title("Accuracy Vs Epochs plot")
plt.xlabel("Accuracy")
plt.ylabel("Loss") 
plt.legend()
plt.show()


