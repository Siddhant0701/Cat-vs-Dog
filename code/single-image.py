import keras
from PIL import Image
import numpy as np
import os

filepath = r'../dataset/new-test/cat/shifaa.jpg'
modelpath = r'../models/cat-or-dog-model-vgg16.h5'
IMG_SIZE= (224,224)

model = keras.models.load_model(modelpath)
results={
    0:'cat',
    1:'dog'
}
count = 0

curr = filepath
im = Image.open(curr).convert('RGB')
im=im.resize(IMG_SIZE)
im=np.expand_dims(im,axis=0)
im=np.array(im)
im=im/255
pred=model.predict([im])

result = np.argmax(pred[0])
print(pred)
print(f'The image is a {results[result]} with {100* max(pred[0][0], pred[0][1])}% confidence')
