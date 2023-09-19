from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import cv2
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network=load_model("my_model.h5")
img = cv2.imread('D:/4.JPG', 0)
img = cv2.resize(img, (28, 28))
dst=255-img
img = dst.reshape((1,28,28,1)).astype('float')/255
predict=network.predict(img)
print(predict)
yuce = np.argmax(predict)
print(yuce)