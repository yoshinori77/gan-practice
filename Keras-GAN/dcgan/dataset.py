import pickle
import matplotlib.pyplot as plt
import sys
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
import os

X_train = []
Y_train = []

X_test = []
Y_test = []

image_dir = '/Users/rasuharu/Dev/gan/image-collector/images/dog/'
i = os.listdir(image_dir)
n = 0
for target_file in i:
    image = (image_dir+target_file)
    temp_img = load_img(image)
    temp_img_array = img_to_array(temp_img)
    print(temp_img_array.shape)
    X_train.append(temp_img_array)
    n = n+1

# np.savez("./gan.npz", x_train=X_train,
#          y_train=Y_train, x_test=X_test, y_test=Y_test)

with open('./gan.p', mode='wb') as f:
    pickle.dump(X_train, f)
