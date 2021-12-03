from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
import numpy as np 

(X_train, y_train) ,(X_test, y_test) = mnist.load_data()

X_train = np.expand_dims(X_train, axis=-1)
X_train = np.repeat(X_train, 3, axis=-1)
X_train = X_train.astype('float32') / 255
# train set / target 
y_train = utils.to_categorical(y_train, num_classes=10)


# validation set / data 
X_test = np.expand_dims(X_test, axis=-1)
X_test = np.repeat(X_test, 3, axis=-1)
X_test = X_test.astype('float32') / 255
# validation set / target 
y_test = utils.to_categorical(y_test, num_classes=10)