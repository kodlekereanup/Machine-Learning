import tensorflow as tf
import matplotlib.pyplot as plt
import h5py

# Import mnist class from datasets module in Keras framework
from tensorflow.keras.datasets import mnist

# Load the data into two tuples.
# Two variables for placing training data and two for placing test data
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

# Printing the length of the training datasets 
# print(len(xTrain), len(yTrain))

# print(xTrain[0])
# print('labels:', yTrain[0])

# plt.imshow(xTrain[0])
# plt.show()

# Normalization -  To reduce computation
#                  To prevent saturation / prevent slow learning 

xTrain = tf.keras.utils.normalize(xTrain, axis = 1)

# Neural Network Model Stucture
model = tf.keras.Sequential()

# Input layer
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# There are 28 x 28 = 784 Neurons in the first layer (input)

# Activation : Relu function is used because of speed increase in training data. Better output in lesser time
# OUTPUT(relu) = max(0, input)
model.add(tf.keras.layers.Dense(100, activation='relu')) #activation = tf.nn.relu

# Number of neurons depends on features 
# Number of layers depends on the complexity of the problem 
# or the amount of features / patterns to be detected

# Second hidden layer
model.add(tf.keras.layers.Dense(100, activation = 'relu'))

# Output layer - Don't use relu here (in this particular question)
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Setting the optimiser function and the loss function
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# NN Structure ends 

model.fit(xTrain, yTrain, epochs=3)

# Overfitting - training a lot of time, like rute learning. The model has memorized the data to give the right results
# only for the trainig data

# The model file contains the matrix with the adjusted weights 

####### TRAINING DONE ########


# TESTING #

valLoss, valAccuracy = model.evaluate(xTest, yTest)
print(f"'validation loss: {valLoss} validation accuracy: {valAccuracy}")

#Save the model and create a model file 
model.save('./models/handdigits.model')
print("Model Saved")