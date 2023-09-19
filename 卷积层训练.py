from keras.utils import to_categorical
from keras import models, layers
from keras.optimizers import RMSprop
from keras.datasets import mnist
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

def LeNet():
    network = models.Sequential()
    network.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    network.add(layers.AveragePooling2D((2, 2)))
    network.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    network.add(layers.AveragePooling2D((2, 2)))
    network.add(layers.Conv2D(filters=120, kernel_size=(3, 3), activation='relu'))
    network.add(layers.Flatten())
    network.add(layers.Dense(84, activation='relu'))
    network.add(layers.Dense(10, activation='softmax'))
    return network
network = LeNet()
network.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float')/255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float')/255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
network.fit(train_images, train_labels, epochs=10, batch_size=128, verbose=2)
test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print("test_loss:", test_loss, "    test_accuracy:", test_accuracy)
network.save('my_model.h5')
