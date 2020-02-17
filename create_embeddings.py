# AUTHOR - DHAIRYA KUMAR


###############################################################################

        #           #           # #  #    #     #            #
      #   #         #           #    #    #     #         #    #
    # ...... #      #           #  # #    # # # #       # ...... #
  #            #    #           #         #     #     #            #
#                #  # # # # #   #         #     #   #                #

###############################################################################

# Importing necessary libraries
import numpy as np
import math
import logging

from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

embeddings_list = []
batch_index = 0
y_train_list = []

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s',level=logging.DEBUG)

def get_embedding(model, face_pixels):
    # Standardising of the face embeddings
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # expand_dims is used to add an additional dimension at the specified axis.
    # If the original shape is (2,2) then expand_dims with axis = 0 will make it (1,2,2)
    samples = expand_dims(face_pixels, axis=0)
    # Making prediction
    yhat = model.predict(samples)
    print(yhat.shape)
    return yhat[0]

if __name__ == '__main__':
    
    model = load_model('model/facenet_keras.h5')
    logging.debug('FaceNet model loaded')

    # Loading the training dataset
    train_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory('data',class_mode='categorical',batch_size=4, target_size = (160,160))
    logging.debug('Dataset loaded')

    logging.debug(train_generator.class_indices)
    num_of_batches = math.ceil(len(train_generator.filenames) / train_generator.batch_size)

    while batch_index < num_of_batches:
        batch_index += 1
        X_train,y_train = train_generator.next()

        for y in y_train:
            y_train_list.append(y)

        for face_pixels in X_train:
            embedding = get_embedding(model, face_pixels)
            embeddings_list.append(embedding)

    # Saving embeddings along with the class labels
    embeddings_list = asarray(embeddings_list)
    savez_compressed('compressed_dataset.npz',embeddings_list,y_train_list)
    logging.debug('Embeddings saved')
