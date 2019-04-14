import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
#%matplotlib inline

from tensorflow.keras.layers import Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

base_model = VGG16(weights='imagenet', include_top = False, 
                   input_tensor=Input(shape=(224, 224, 3)))

batch_size = 32
nrow = 224
ncol = 224
nchan = 3

batch_shape = (batch_size, nrow, ncol, nchan)
x = np.zeros(batch_shape)

model = Sequential()
for i in base_model.layers:
    model.add(i)
for i in model.layers:
    i.trainable = False
    
model.add(Flatten(input_shape = (224, 224, 3)))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(24, activation = 'softmax'))

train_data_dir = './Dataset'
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
                        train_data_dir,
                        target_size=(nrow,ncol),
                        batch_size=batch_size,
                        class_mode='categorical')

test_data_dir = './Val'
batch_size = 32
test_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_generator = test_datagen.flow_from_directory(
                        test_data_dir,
                        target_size=(nrow,ncol),
                        batch_size=batch_size,
                        class_mode='categorical')


model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
#categorical_crossentropy
steps_per_epoch =  train_generator.n // batch_size
validation_steps =  test_generator.n // batch_size

nepochs = 10

hist = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=nepochs,
    validation_data=test_generator,
    validation_steps=validation_steps)

#Model to JSON
model_json = model.to_json()
with open("model.json", "w") as f:
    f.write(model_json)
#Weights to HDF5
model.save_weights("weights.h5")
print("Saved model to disk")

with open('model.json','r') as f:
    json = f.read()
loaded_model = model_from_json(json)
loaded_model.load_weights("weights.h5", by_name=True)

 
# evaluate loaded model on test data
loaded_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

Xtest,ytest = test_generator.next()
yhat = loaded_model.predict(Xtest)
l = len(Xtest)
for i in range (l):
    disp_image(Xtest[i])
    plt.title("Actual: "+str(ytest[i])+" Predicted: " + str(yhat[i]))
    plt.show()