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

from simulation_cobinding_2 import get_simulated_dataset

from helper_funcs import MetricsCallback


import matplotlib.pyplot as plt

from sklearn.manifold import TSNE



def CAC(input_shape=(1,1024,4)):

    input_layer = Input(shape=input_shape)

    x = Conv2D(32, 15, strides=4, padding='same', activation='relu', name='conv1', input_shape=input_shape)(input_layer)

    x = Conv2D(64, 15, strides=4, padding='same', activation='relu', name='conv2', input_shape=input_shape)(x)

    x = Conv2D(128, 11, strides=4, padding='same', activation='relu', name='conv3', input_shape=input_shape)(x)

    x = Flatten()(x)

    encoded = Dense(units=10, name='embedding')(x)

    ###

    x = Dense(units=2048, activation='relu')(encoded)

    x = Reshape(  (1, 16, 128)   )(x)

    x = Conv2DTranspose(64, 15, strides=(1,4), padding='same', activation='relu', name='deconv3')(x)

    x = Conv2DTranspose(32, 15, strides=(1,4), padding='same', activation='relu', name='deconv2')(x)

    decoded = Conv2DTranspose(4, 15, strides=(1,4), padding='same', name='deconv1')(x)

    ###

    autoencoder = Model(input_layer, decoded)

    autoencoder.summary()

    encoder = Model(input_layer, encoded, name='encoder')

    encoder.summary()

    simutation_parameters = {
        "PWM_file_1": "./MA0835.1.jaspar",
        "PWM_file_2": "./MA0515.1.jaspar",
        "seq_length":1024,
        "center_pos": 50,
        "interspace":10
    }
    
    [train_X, train_Y, test_X , test_Y] =  get_simulated_dataset(parameters = simutation_parameters, train_size = 20000, test_size = 5000)

    print(train_X.shape)

    #################################
    # build autoencoder model

    
 
    autoencoder.compile(optimizer='adam',loss='mse')

    history_autoencoder=autoencoder.fit(x=train_X,
                                  y=train_X,
                                  batch_size=64,
                                  epochs=10,
                                  verbose=1,
                                  callbacks=[ History()],
                                  validation_data=(test_X, test_X))

    encoded_imgs = encoder.predict(test_X)
    
    print(encoded_imgs.shape)

    colors = ['#e41a1c', '#377eb8', '#4daf4a']

    X_embedded = TSNE(n_components=2).fit_transform(encoded_imgs)

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c = np.array(colors)[test_Y.flatten()]   )
    plt.colorbar()
    plt.show()



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


def CAC_2(input_shape=(1,1024,4)):

    
    model = Sequential( )

    model.add(Conv2D(5, 11, strides=1, padding='same', activation='relu', name='conv1', input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(1,4)))

    model.add(Conv2D(5, 11, strides=1, padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(1,4)))

    model.add(Conv2D(5, 11, strides=1, padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(1,4)))

    model.add(Flatten())

    model.add(Dense(units = 10))

    model.add(Dense(units=80, activation='relu'))

    model.add(Reshape(  (1, 16, 5)   ))

    model.add(UpSampling2D(size=(1,4)))

    model.add(  Conv2DTranspose(5, 11, strides=(1,1), padding='same', activation='relu', name='deconv3') )

    model.add(UpSampling2D(size=(1,4)))

    model.add(  Conv2DTranspose(5, 11, strides=(1,1), padding='same', activation='relu') )

    model.add(UpSampling2D(size=(1,4)))

    model.add(Conv2DTranspose(4, 11, strides=(1,1), padding='same', activation='relu'))

    model.summary()
    
    return 0 
    

    '''
    input_layer = Input(shape=input_shape)

    x = Conv2D(5, 11, strides=1, padding='same', activation='relu', name='conv1', input_shape=input_shape)(input_layer)
    
    x=MaxPooling2D(pool_size=(1,16))(x)

    x = Flatten()(x)

    encoded = Dense(units = 10)(x)

    x = Dense(units=320, activation='relu')(encoded)

    x = Reshape(  (1, 64, 5)   )(x)

    x=UpSampling2D(size=(1,16))(x)

    decoded = Conv2DTranspose(4, 11, strides=(1,1), padding='same', activation='relu', name='deconv3')(x)

    autoencoder = Model(input_layer, decoded)

    autoencoder.summary()

    encoder = Model(input_layer, encoded, name='encoder')
    '''

    '''
    input_layer = Input(shape=input_shape)

    x = Conv2D(5, 11, strides=4, padding='same', activation='relu', name='conv1', input_shape=input_shape)(input_layer)

    x = Conv2D(64, 15, strides=4, padding='same', activation='relu', name='conv2', input_shape=input_shape)(x)

    x = Conv2D(128, 11, strides=4, padding='same', activation='relu', name='conv3', input_shape=input_shape)(x)

    x = Flatten()(x)

    encoded = Dense(units=10, name='embedding')(x)

    ###

    x = Dense(units=2048, activation='relu')(encoded)

    x = Reshape(  (1, 16, 128)   )(x)

    x = Conv2DTranspose(64, 15, strides=(1,4), padding='same', activation='relu', name='deconv3')(x)

    x = Conv2DTranspose(32, 15, strides=(1,4), padding='same', activation='relu', name='deconv2')(x)

    decoded = Conv2DTranspose(4, 15, strides=(1,4), padding='same', name='deconv1')(x)

    ###

    autoencoder = Model(input_layer, decoded)

    autoencoder.summary()

    encoder = Model(input_layer, encoded, name='encoder')

    encoder.summary()
    '''


    simutation_parameters = {
        "PWM_file_1": "./JASPAR/MA0835.1.jaspar",
        "PWM_file_2": "./JASPAR/MA0515.1.jaspar",
        "seq_length":1024,
        "center_pos": 100,
        "interspace":10
    }
    
    [train_X, train_Y, test_X , test_Y] =  get_simulated_dataset(parameters = simutation_parameters, train_size = 20000, test_size = 5000)

    print(train_X.shape)
    
    #################################
    # build autoencoder model

    
 
    autoencoder.compile(optimizer='adam',loss='mse')

    history_autoencoder=autoencoder.fit(x=train_X,
                                  y=train_X,
                                  batch_size=32,
                                  epochs=20,
                                  verbose=1,
                                  callbacks=[ History()],
                                  validation_data=(test_X, test_X))


    encoded_imgs = encoder.predict(test_X)
    
    print(encoded_imgs.shape)

    colors = ['#e41a1c', '#377eb8', '#4daf4a']

    X_embedded = TSNE(n_components=2).fit_transform(encoded_imgs)

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c = np.array(colors)[test_Y.flatten()]   )
    plt.colorbar()
    plt.show()
   
if __name__ == "__main__":
    #example_Cov_auto()
    CAC_2()




 

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


'''

'''

    model = Sequential( )

    model.add(Conv2D(5, 11, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))

    model.add(Flatten())

    model.add(Dense(units = 5))

    model.add(Dense(units=160, activation='relu'))

    model.add(Reshape(  (1, 32, 5)   ))

    model.add(  Conv2DTranspose(4, 11, strides=(1,2), padding='same', activation='relu', name='deconv3') )

    model.summary()
'''