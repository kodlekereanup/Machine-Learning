import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import numpy as np

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

xTest = tf.keras.utils.normalize(xTest, axis=1)
model = load_model('models/handdigits.model')

# Use this model to predict the output

predictions = model.predict([xTest])

print(predictions[0])
print("Predicted value is:", np.argmax(predictions[0]))

plt.imshow(xTest[0])
plt.show()

