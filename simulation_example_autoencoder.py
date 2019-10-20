from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
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

from simulation_cobinding import get_simulated_dataset

from helper_funcs import MetricsCallback



def example_Cov_auto():
    simutation_parameters = {
        "PWM_file_1": "/home/qan/Desktop/DeepEpitif/DeepMetif/JASPAR2018_CORE_vertebrates_non-redundant_pfms_jaspar/MA0835.1.jaspar",
        "PWM_file_2": "/home/qan/Desktop/DeepEpitif/DeepMetif/JASPAR2018_CORE_vertebrates_non-redundant_pfms_jaspar/MA0515.1.jaspar",
        "seq_length":1000,
        "center_pos": 100,
        "interspace":50
    }
    
    [train_X, train_Y, test_X , test_Y] =  get_simulated_dataset(parameters = simutation_parameters, 
    train_size = 1000, test_size = 1000)

    print(train_X.shape[1::])


    
    #################################
    # build autoencoder model

    Cov_auto = Sequential()
    Cov_auto.add(Conv2D(filters=5,kernel_size=(1,15), activation="relu", padding="same",input_shape=(1,1000,4) ))
    Cov_auto.add(MaxPooling2D(   pool_size=(1,2)   ) )
    
    #Cov_auto.add(Flatten())
    #Cov_auto.add(Flatten())

    Cov_auto.add(Conv2D(filters=5,kernel_size=(1,15), activation="relu", padding="same" ))
    Cov_auto.add(UpSampling2D( size=(1,2)  ))
    Cov_auto.add(Conv2D(filters=4,kernel_size=(1,15), activation="sigmoid", padding="same"))

    Cov_auto.summary()

    Cov_auto.compile(optimizer='adam',loss='binary_crossentropy')

    # metrics_callback=MetricsCallback(train_data=(train_X,train_Y), validation_data=(valid_X,valid_Y))

    history_autoencoder=Cov_auto.fit(x=train_X,
                                  y=train_X,
                                  batch_size=10,
                                  epochs=50,
                                  verbose=1,
                                  callbacks=[ History()],
                                  validation_data=(test_X, test_X))


if __name__ == "__main__":
    example_Cov_auto()