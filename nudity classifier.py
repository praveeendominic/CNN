import tensorflow as tf
from tensorflow import keras
# from keras_preprocessing import image
# from keras.preprocessing.image import ImageDataGenerator
# import tensorflow_hub as hub


# import sys
# from PIL import Image

def load_data():
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.10, shear_range= 0.10)
    train_data = train_gen.flow_from_directory("data/train",
                                               class_mode='categorical',
                                               batch_size=32,
                                               target_size =(128,128))

    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.10, shear_range= 0.10)
    test_data = train_gen.flow_from_directory("data/test",
                                               class_mode='categorical',
                                               batch_size=32,
                                               target_size =(128,128))
    return (train_data, test_data)


# def load_data():
#     import pathlib
#     import cv2
#     import numpy as np
#
#     dir=pathlib.Path("data1")
#
#     image_dict = {"nude": dir.glob("nude/*.jpeg"),
#                   "safe": dir.glob("safe/*.jpeg"),
#                   "sexy": dir.glob("sexy/*.jpeg")
#                   }
#
#     label_dict = {"nude": 0,
#                   "safe": 1,
#                   "sexy": 2
#                   }
#
#     X,y = [], []
#     for categ_key, categ_files in image_dict.items():
#         for val in categ_files:
#             im=cv2.imread(str(val))
#             im=cv2.resize(im,(128,128))#resize
#             im=im/255 #rescale
#             # print(im)
#
#             X.append(im)
#             y.append(label_dict.get(categ_key))
#
#     X=np.array(X)
#     y=np.array(y)
#
#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
#
#     return {"X_train":X_train,
#             "X_test": X_test,
#             "y_train": y_train,
#             "y_test": y_test}


def instantiate_model():
    import tensorflow_hub as hub
    feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    pretrained_model_without_top_layer = hub.KerasLayer(feature_extractor_model, input_shape=(128, 128, 3), trainable=False)


    LAYERS = [
        pretrained_model_without_top_layer,
        tf.keras.layers.Conv2D( kernel_size=3, filters=8, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2)),

        tf.keras.layers.Conv2D(kernel_size=3, filters=16, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2)),

        tf.keras.layers.Conv2D(kernel_size=3, filters=32, activation='relu', padding='valid'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(kernel_size=3, filters=64, activation='relu', padding='valid'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=(2, 2)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(kernel_size=3, filters=64, activation='relu', padding='valid'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=3, activation='softmax')
    ]

    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=1e-3)
    LOSS = tf.keras.losses.sparse_categorical_crossentropy
    METRICS = ['accuracy']

    cnn = tf.keras.models.Sequential(LAYERS)
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return cnn


def train_model(model, X_train, y_train):
    history=model.fit(x=train_data, validation_data=test_data, epochs=30)
    # history=model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.10, verbose=2)

    return model


model=instantiate_model()
print(model.summary())

# res= load_data()
# X_train, X_test, y_train, y_test = res.get("X_train"), res.get("X_test"), res.get("y_train"), res.get("y_test")

train_data, test_data=load_data()

trained_model=train_model(model,train_data, test_data)

from keras_preprocessing import image
import numpy as np
test_image=np.expand_dims(image.img_to_array(image.load_img("data/test/sexy/0A2EBEF5-7E51-4A45-8EA1-4363798A2ADA.jpg", target_size=(128,128))), axis=0)

print(train_data.class_indices)

print(trained_model.predict(test_image)[0][0])

print()




