import tensorflow as tf

import numpy as np
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

for i,c in enumerate(celsius_q):
  print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

# create dense neuron layer with one neuron, input_shape=[1] means one dimensional array
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

# define model with above layer
model = tf.keras.Sequential([l0])

# compile model with loss and optimizer functions
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

# train model
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")

# draw plot loss vs. epochs
import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()

# make prediction for 100 degree celsius
print("Prediction for {} degree Celsius is: {}".format(100,model.predict([100.0])))

# print layer variables
print("These are the layer variables: {}".format(l0.get_weights()))