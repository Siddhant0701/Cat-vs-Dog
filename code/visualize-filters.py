import numpy as np
from keras.models import load_model, Sequential
import cv2

model = load_model(r'../models/cat-or-dog-model-vgg.h5')

#Select a convolutional layer
layer = model.layers[0]

#Get weights
kernels, biases = layer.get_weights()

#Normalize kernels into [0, 1] range for proper visualization
print(kernels.shape[3])
for i in range(kernels.shape[3]):     
    kernels[:,:,:,i] = (kernels[:,:,:,i] - np.min(kernels, axis=3)) / np.max(kernels, axis=3) - np.min(kernels, axis=3)

#Weights are usually (width, height, channels, num_filters)
#Save weight images

for i in range(kernels.shape[3]):
    filter = kernels[:, :, :, i]
    cv2.imwrite('filter-{}.png'.format(i), filter)
