from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import MobileNetV2
import tensorflow as tf
from tensorflow.keras import Sequential

def simple_model(input_shape = (784,), num_classes= 10):
    inputs = Input(shape = input_shape)
    x = layers.Dense(16, activation = "relu")(inputs)
    x = layers.Dense(32,activation = "relu")(x)
    x = layers.Dense(64,activation = "relu")(x)
    outputs = layers.Dense(num_classes)(x)

    return Model(inputs = inputs, outputs = outputs, name = "simple_model")

def model_mobilnet(input_shape = (None,784,1), num_classes = 10):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                               include_top=False,
                                               weights='imagenet')
    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(num_classes)

    inputs = tf.keras.Input(shape=(input_shape))

    x = base_model(inputs, training=False)
    x = global_average_layer(x)
    x = layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return model

    



