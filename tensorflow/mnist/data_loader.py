from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
import numpy as np 

(X_train, y_train) ,(X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784).astype("float32") / 255
X_test = X_test.reshape(10000, 784).astype("float32") / 255

# train set / target 
y_train = utils.to_categorical(y_train, num_classes=10) 
y_test = utils.to_categorical(y_test, num_classes=10)