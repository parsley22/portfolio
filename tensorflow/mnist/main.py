from model import simple_model
from tensorflow.keras import losses,metrics, optimizers
from data_loader import X_train, y_train ,X_test, y_test

# x_train.shape, y_train.shape: (60000, 28, 28) (60000,)
# x_test.shape,  y_test.shape : (10000, 28, 28) (10000,)


model = simple_model(10)

model.compile(
              loss = losses.CategoricalCrossentropy(),
              metrics = metrics.CategoricalAccuracy(),
              optimizer = optimizers.Adam()
              )

model.fit(X_train, y_train, batch_size = 32, epochs = 2)
