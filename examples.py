from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.python.keras.optimizers import Adadelta, SGD, RMSprop
import tensorflow.python.keras.losses
from tensorflow.python.keras.constraints import maxnorm
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.regularizers import l1, l2
from tensorflow.python.keras.callbacks import EarlyStopping, History
from tensorflow.python.keras import backend as K 
from tensorflow.python.keras.models import model_from_json


K.set_image_data_format('channels_last')

from random import choice, seed
import numpy as np

from simulations import get_simulated_dataset

from helper_funcs import MetricsCallback

from interpret import plot_CNN_filters


#Define the model architecture in keras
def example_1():

    simutation_parameters = {
        "PWM_file": "/home/qan/Desktop/DeepEpitif/DeepMetif/JASPAR2018_CORE_vertebrates_non-redundant_pfms_jaspar/MA0835.1.jaspar",
        "seq_length":100,
        "center_pos": 20,
        "motif_width":14,
        "metif_level" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    }
    
    [train_X, train_Y, valid_X, valid_Y, test_X , test_Y] =  get_simulated_dataset(parameters = simutation_parameters, train_size = 16000, valid_size = 2000, test_size = 20)

    
    #print(train_X.dtype)
    #print(train_Y.dtype)
    #print(train_X[2,:,:,:])
    #print(train_Y)
    #print(train_X.shape[1::])
    
    #exit()
    one_filter_keras_model=Sequential() 
    one_filter_keras_model.add(Conv2D(filters=5,kernel_size=(1,15),padding="same",input_shape=  train_X.shape[1::]  ))
    one_filter_keras_model.add(BatchNormalization(axis=-1))
    one_filter_keras_model.add(Activation('relu'))
    one_filter_keras_model.add(MaxPooling2D(pool_size=(1,35)))
    one_filter_keras_model.add(Flatten())
    one_filter_keras_model.add(Dense(1))
    one_filter_keras_model.add(Activation("sigmoid"))
    one_filter_keras_model.summary()

    one_filter_keras_model.compile(optimizer='adam',loss='binary_crossentropy')


    metrics_callback=MetricsCallback(train_data=(train_X,train_Y),
                                 validation_data=(valid_X,valid_Y))

    
    print(one_filter_keras_model.get_weights())

    history_one_filter=one_filter_keras_model.fit(x=train_X,
                                  y=train_Y,
                                  batch_size=10,
                                  epochs=50,
                                  verbose=1,
                                  callbacks=[ History(),metrics_callback],
                                  validation_data=(valid_X,
                                                   valid_Y))
    #print(one_filter_keras_model.get_weights())

    one_filter_keras_model_json = one_filter_keras_model.to_json()
    with open("one_filter_keras_model.json", "w") as json_file:
        json_file.write(one_filter_keras_model_json)

    one_filter_keras_model.save_weights("one_filter_keras_model.h5")
    print("Saved model to disk")

    #one_filter_keras_model.save(filepath="one_filter_keras_model.h5", overwrite=True)

# visualize CNN filters
def example_2():
    # load json and create model
    json_file = open('one_filter_keras_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("one_filter_keras_model.h5")
    print("Loaded model from disk")

    CNN_layer_weights = (loaded_model.get_weights())[0]
    #print(CNN_layer_weights)

    plot_CNN_filters(CNN_layer_weights)

# interpret CNN using packages deeplift
def example_3():
    pass





if __name__ == "__main__":
    #seed(1993)
    #np.random.seed(1993)
    example_1()
    #example_2()