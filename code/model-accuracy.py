import keras
from PIL import Image
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

test_dir = r'dataset/test-set/'
model_name = 'cat-or-dog-model-vgg.h5'
model_path = r'../models/' + model_name
IMG_SIZE= (128,128)


model = keras.models.load_model(model_path)
model_classes={
    0:'cats',
    1:'dogs'
}

datagen = ImageDataGenerator(rescale=1./255)
data = datagen.flow_from_directory(test_dir, target_size=IMG_SIZE, shuffle = False, color_mode='rgb', class_mode='categorical')


probabilities= model.predict(data)
results = np.argmax(probabilities, axis = 1)

print('Classification Report')
target_names = data.classes
class_labels = ['cats', 'dogs']  
report = classification_report(target_names, results, target_names=class_labels)
print(report) 
