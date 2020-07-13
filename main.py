import tensorflow as tf
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import random 

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
plt.imshow(x_train[0], cmap = 'gray')

#view data
l_grid = w_grid = 15
fig, axes = plt.subplots(l_grid, w_grid, figsize = (17, 17))
axes = axes.ravel()
n_training = len(x_train)

for i in np.arange(0, w_grid * l_grid):
    index = np.random.randint(0, n_training)
    axes[i].imshow(x_train[index])
    axes[i].set_title(y_train[index], fontsize = 8)
    axes[i].axis('off')
    
    
x_train = x_train / 255.0
y_train = y_train / 255.0

#add noise
noise_factor = 0.3
noise_dataset = []
for img in x_train:
    noisy_image = img + noise_factor * np.random.randn(*img.shape)
    noisy_image = np.clip(noisy_image, 0, 1)
    noise_dataset.append(noisy_image)
    
#noise test set
noise_dataset = np.array(noise_dataset)
noise_factor = 0.3
noise_test_set = []
for img in x_test:
    noisy_image = img + noise_factor * np.random.randn(*img.shape)
    noisy_image = np.clip(noisy_image, 0, 1)
    noise_test_set.append(noisy_image)
    
noise_test_set = np.array(noise_test_set)
plt.imshow(noise_dataset[69], cmap = 'gray')

#autoencoder architecture
from tensorflow.keras.models import Sequential
autoencoder = Sequential()

#encoder
input_img = tf.keras.layers.Input(shape=(28, 28, 1))
autoencoder.add(tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, strides = 2, padding = 'same', input_shape = (28, 28, 1)))
autoencoder.add(tf.keras.layers.Conv2D(filters = 8, kernel_size = 3, strides = 2, padding = 'same'))
autoencoder.add(tf.keras.layers.Conv2D(filters = 8, kernel_size = 3, strides = 1, padding = 'same'))     
                
#decoder
autoencoder.add(tf.keras.layers.Conv2DTranspose(filters = 16, kernel_size = 3, strides = 2, padding = 'same'))
autoencoder.add(tf.keras.layers.Conv2DTranspose(filters = 1, kernel_size = 3, strides = 2,activation = 'sigmoid', padding = 'same'))

#see model
autoencoder.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.Adam(lr = 0.01))
autoencoder.summary()

#train
autoencoder.fit(noise_dataset.reshape(-1, 28, 28, 1),
               x_train.reshape(-1, 28, 28, 1),
               epochs = 2,
               batch_size = 20,
               validation_data = (noise_test_set.reshape(-1, 28, 28, 1), x_test.reshape(-1, 28, 28, 1)))
           
#accuracy
evaluation = autoencoder.evaluate(noise_test_set.reshape(-1, 28, 28, 1), x_test.reshape(-1, 28, 28, 1))
print('Test Loss : {:.3f}'.format(evaluation))

#view output images
predicted = autoencoder.predict(noise_test_set[:10].reshape(-1, 28, 28, 1))
fig, axes  = plt.subplots(nrows = 2, ncols = 10, sharex = True, sharey = True, figsize = (20, 4))
for images, row in zip([noise_test_set[:10], predicted], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)), cmap = 'Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
