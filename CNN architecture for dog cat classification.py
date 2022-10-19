import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator

traingen=ImageDataGenerator(rescale=1./255,
                         horizontal_flip=True,
                         shear_range=0.2,
                         zoom_range=0.2)
testgen=ImageDataGenerator(rescale=1./255)

training_data=traingen.flow_from_directory('dataset/training_set',
                             class_mode='binary',
                             batch_size=32,
                             target_size=(64,64))

test_data= testgen.flow_from_directory("dataset/test_set",
                                       class_mode='binary',
                                       batch_size=32,
                                       target_size=(64,64))


#Initialize the CNN
LAYERS = [Conv2D(filters=32,kernel_size=3, padding='valid',
                 input_shape=(64,64,3), activation='relu'),
          MaxPool2D(pool_size=2, strides=(2,2)),
          Conv2D(filters=64,kernel_size=3, padding='valid',
                 activation='relu'),
          MaxPool2D(pool_size=2, strides=(2,2)),
          Conv2D(filters=128, kernel_size=3, padding='valid',
                 activation='relu'),
          MaxPool2D(pool_size=2, strides=(2, 2)),
          Flatten(),
          Dense(units=128, activation='relu'),
          Dense(1, activation='sigmoid')
          ] #2*2 pool_size is optimal

model = Sequential(LAYERS)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

import sys
from PIL import Image
sys.modules['Image'] = Image

model.fit(x=training_data,validation_data=test_data,epochs=25)




