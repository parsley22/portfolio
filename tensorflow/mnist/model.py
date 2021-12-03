from tensorflow.keras import layers
import tensorflow

class simple_model(tensorflow.keras.Model):

    def __init__(self, num_classes):
        super(simple_model, self).__init__()
        
        self.num_classes = num_classes

        self.d1 = layers.Dense(16)
        self.d2 = layers.Dense(32)
        self.d3 = layers.Dense(num_classes)

    def call(self, input_tensor):

        x = self.d1(input_tensor)
        x = self.d2(x)
        x = self.d3(x)

        return x

