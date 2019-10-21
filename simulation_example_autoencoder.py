from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten

from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D

from tensorflow.python.keras.layers import Input, Conv2DTranspose

from tensorflow.python.keras.optimizers import Adadelta, SGD, RMSprop
import tensorflow.python.keras.losses
from tensorflow.python.keras.constraints import maxnorm
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.regularizers import l1, l2
from tensorflow.python.keras.callbacks import EarlyStopping, History
from tensorflow.python.keras import backend as K 
from tensorflow.python.keras.models import model_from_json, Model


K.set_image_data_format('channels_last')

from random import choice, seed
import numpy as np

from simulation_cobinding import get_simulated_dataset

from helper_funcs import MetricsCallback


import matplotlib.pyplot as plt



def CAC(input_shape=(1,1024,4), filters = [32, 64, 128 , 10]):
    model = Sequential( )

    model.add(Conv2D(32, 15, strides=4, padding='same', activation='relu', name='conv1', input_shape=input_shape))

    model.add(Conv2D(64, 15, strides=4, padding='same', activation='relu', name='conv2', input_shape=input_shape))

    model.add(Conv2D(128, 11, strides=4, padding='same', activation='relu', name='conv3', input_shape=input_shape))

    model.add(Flatten())

    model.add(Dense(units=10, name='embedding'))

    model.add(Dense(units=2048, activation='relu'))

    model.add(Reshape(  (1, 16, 128)   ))

    model.add(Conv2DTranspose(64, 15, strides=(1,4), padding='same', activation='relu', name='deconv3'))

    model.add(Conv2DTranspose(32, 15, strides=(1,4), padding='same', activation='relu', name='deconv2'))

    model.add(Conv2DTranspose(4, 15, strides=(1,4), padding='same', name='deconv1'))
    
    model.summary()
    return model



def example_Cov_auto():
    simutation_parameters = {
        "PWM_file_1": "/home/qan/Desktop/DeepEpitif/DeepMetif/JASPAR2018_CORE_vertebrates_non-redundant_pfms_jaspar/MA0835.1.jaspar",
        "PWM_file_2": "/home/qan/Desktop/DeepEpitif/DeepMetif/JASPAR2018_CORE_vertebrates_non-redundant_pfms_jaspar/MA0515.1.jaspar",
        "seq_length":1024,
        "center_pos": 50,
        "interspace":10
    }
    
    [train_X, train_Y, test_X , test_Y] =  get_simulated_dataset(parameters = simutation_parameters, train_size = 20000, test_size = 5000)

    


    
    #################################
    # build autoencoder model

    model = CAC()
 
    model.compile(optimizer='adam',loss='mse')

    history_autoencoder=model.fit(x=train_X,
                                  y=train_X,
                                  batch_size=10,
                                  epochs=20,
                                  verbose=1,
                                  callbacks=[ History()],
                                  validation_data=(test_X, test_X))


    

'''
   input_shape = (1, 1000, 4)

    latent_dim = 20

    inputs = Input(shape=input_shape, name='encoder_input')
    
    x=inputs

    x = Conv2D(filters=5,kernel_size=(1,15), activation="relu", padding="same" )(x)

    #x = MaxPooling2D(   pool_size=(1,2)   )(x)

    shape = K.int_shape(x)

    x=Flatten()(x)

    latent = Dense(latent_dim, name='latent_vector')(x)

    encoder = Model(inputs, latent, name='encoder')

    encoder.summary()

    # build decoder model

    latent_inputs = Input(shape=(latent_dim, ) , name='decoder_input' )

    x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)

    x=Reshape((shape[1], shape[2], shape[3]))(x)

    #x = UpSampling2D( size=(1,2)  )(x)

    #x = Conv2DTranspose(filters=5,kernel_size=(1,15), activation="relu", padding="same" )(x)

    x = Conv2DTranspose(filters=4,
                    kernel_size=(1,15),
                    padding='same')(x)


    outputs = Activation('sigmoid', name='decoder_output')(x)


    decoder = Model(latent_inputs, outputs, name='decoder')
    
    decoder.summary()

    autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    
    autoencoder.summary()

    autoencoder.compile(optimizer='adam',loss='mse')

        encoded_imgs = encoder.predict(test_X)
    
    print(encoded_imgs.shape)

    colors = ['#e41a1c', '#377eb8', '#4daf4a']

    plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c = np.array(colors)[test_Y.flatten()]   )
    plt.colorbar()
    plt.show()

'''

if __name__ == "__main__":
    example_Cov_auto()
    #CAC()

