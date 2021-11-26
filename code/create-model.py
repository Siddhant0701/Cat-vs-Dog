import numpy as np
import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,\
     Dropout,Flatten,Dense,Activation,\
     BatchNormalization

from tensorflow.keras.optimizers import Adam


HEIGHT = 128
WIDTH = 128
CHANNELS = 3
IMG_SIZE = (HEIGHT,WIDTH)

EPOCHS = 20
BATCH_SIZE = 10
LR = 0.0001
SPLIT = 0.2
STEPS = (8000*(1-SPLIT))//BATCH_SIZE
VALIDATION_STEPS = (8000*(SPLIT))//BATCH_SIZE

train_dir = r'../dataset/training-set'


## Data Generators
train_datagen = ImageDataGenerator (rescale=1./255, validation_split=SPLIT, 
                                    vertical_flip=True, horizontal_flip=True,
                                    rotation_range = 5, zoom_range= 0.1,
                                    shear_range=0.2)

train_data = train_datagen.flow_from_directory( train_dir, target_size=IMG_SIZE, shuffle=True, batch_size = BATCH_SIZE, 
                                                class_mode='categorical', subset = 'training', color_mode='rgb')

validation_data = train_datagen.flow_from_directory(train_dir, target_size=IMG_SIZE, shuffle=True, batch_size = BATCH_SIZE, 
                                                    class_mode='categorical', subset = 'validation', color_mode='rgb')

## Sequential Model
model  = Sequential([
    Conv2D(64, kernel_size = (3,3), activation='relu', input_shape = (WIDTH, HEIGHT, CHANNELS)),
    MaxPooling2D(pool_size = (2,2), strides = 2),

    Conv2D(128, kernel_size = (3,3), activation='relu'),
    MaxPooling2D(pool_size = (2,2), strides=2),

    Conv2D(256, kernel_size = (3,3), activation='relu'),
    MaxPooling2D(pool_size =(2,2), strides=2),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),

    Dense(2, activation='softmax')
])


## Compiling the model
opt = Adam(learning_rate=LR)
model.compile(optimizer=opt, loss = 'categorical_crossentropy', metrics=['accuracy'])

## Summary
model.summary()

model.fit(train_data, steps_per_epoch= STEPS, validation_data= validation_data, validation_steps= VALIDATION_STEPS, epochs=EPOCHS, verbose = 2)

print(train_data.class_indices)
model.save("../models/cat-or-dog-model-vgg.h5")
