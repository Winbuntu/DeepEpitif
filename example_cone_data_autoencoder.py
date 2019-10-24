from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.python.keras.optimizers import Adadelta, SGD, RMSprop
import tensorflow.python.keras.losses
from tensorflow.python.keras.constraints import maxnorm
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.regularizers import l1, l2
from tensorflow.python.keras.callbacks import EarlyStopping, History, ModelCheckpoint
from tensorflow.python.keras import backend as K 
from tensorflow.python.keras.models import model_from_json, Model
from tensorflow.python.keras.layers import Input, Conv2DTranspose

import pandas as pd

K.set_image_data_format('channels_last')

from random import choice, seed
import numpy as np

from simulations import get_simulated_dataset

from helper_funcs import *

from interpret import plot_CNN_filters



import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


from generator_for_autoencoder import DataGenerator,separate_dataset

from runtime_metrics import *


from numpy.random import seed
seed(1234)
from tensorflow import set_random_seed
set_random_seed(1234)

def initialize_model():

    model = Sequential()
    model.add(Conv2D(40, 11, strides=1, padding='same', input_shape=(1,1024,4)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(Conv2D(40, 11, strides=1, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size = (1,64) ))

    model.add(Flatten())

    model.add(Dense(units=500))

    model.add(Dense(units=640))

    model.add(Reshape((1,16,40)))

    model.add(Conv2DTranspose(40, 11, strides=(1,64), padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(40, 11, strides=(1,1), padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(Conv2D(4, 11, strides=1, padding='same', activation='sigmoid'))
    model.summary()
    model.compile(optimizer='adam',loss='mse')

    return model


class CustomCheckpoint(Callback):
    def __init__(self, filepath, encoder):
        self.monitor = 'val_loss'
        self.monitor_op = np.less
        self.best = np.Inf

        self.filepath = filepath
        self.encoder = encoder

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if self.monitor_op(current, self.best):
            self.best = current
            # self.encoder.save_weights(self.filepath, overwrite=True)
            self.encoder.save(self.filepath, overwrite=True) # Whichever you prefer


def example_generator():
    
    separate_dataset("True_target_with_labels_128.bed", ["chr1"], "valid.bed")
    separate_dataset("True_target_with_labels_128.bed", ["chr2","chr19"], "test.bed")
    separate_dataset("True_target_with_labels_128.bed", ["chr3","chr4","chr5","chr6","chr7","chr8",
                                                            "chr9","chr10","chr11","chr12","chr13",
                                                            "chr14","chr15","chr16","chr17",
                                                            "chr18","chr20","chr21","chr22"], "train.bed")
                                                            
    

    train_gen = DataGenerator(data_path="train.bed", 
    ref_fasta = "../GSM1865005_allC.MethylC-seq_WT_rods_rep1.tsv/GRCm38.primary_assembly.genome.fa.gz",
    genome_size_file="./mm10.genome.size", epi_track_files=None,
    tasks=["TARGET"],upsample=False)
    
    valid_gen = DataGenerator(data_path="valid.bed", 
    ref_fasta = "../GSM1865005_allC.MethylC-seq_WT_rods_rep1.tsv/GRCm38.primary_assembly.genome.fa.gz",
    genome_size_file="./mm10.genome.size", epi_track_files=None,
    tasks=["TARGET"],upsample=False)

    #model = initialize_model()
    # add functional models here

    input_shape = (1,128,4)
    input_layer = Input(shape=input_shape)
    x = Conv2D(40, 11, strides=1, padding='same', input_shape=input_shape)(input_layer)
    x = BatchNormalization(axis= -1 )(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(1,32))(x)

    encoded = Flatten()(x)

    x = Reshape( (1,4,40) )(encoded)

    x = UpSampling2D(size=(1,32))(x)

    decoded = Conv2D(4,11,padding='same', activation='sigmoid')(x)

    ###

    autoencoder = Model(input_layer, decoded)

    autoencoder.summary()

    encoder = Model(input_layer, encoded, name='encoder')

    encoder.summary()

    
    autoencoder.compile(optimizer='adam',loss='mse')
    encoder.compile(optimizer='adam',loss='mse')


    trainning_history=autoencoder.fit_generator(train_gen,
                                                  validation_data=valid_gen,
                                                  #steps_per_epoch=5000,
                                                  #validation_steps=500,
                                                  epochs=600,
                                                  verbose=1,
                                                  use_multiprocessing=True,
                                                  workers=6,
                                                  max_queue_size=200,
                callbacks=[History(), ModelCheckpoint("ATAC_peak_autoencoder_32.h5", 
                                           monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False) , 
                                           CustomCheckpoint('ATAC_peak_encoder_32.h5', encoder)])



def prediction_and_evaluation():
    from tensorflow.python.keras.models import load_model

    encoder_loaded = load_model("ATAC_peak_encoder_32.h5")

    autoencoder_loaded = load_model("ATAC_peak_autoencoder_32.h5")

    test_gen = DataGenerator(data_path="test.bed", 
    ref_fasta = "../GSM1865005_allC.MethylC-seq_WT_rods_rep1.tsv/GRCm38.primary_assembly.genome.fa.gz",
    genome_size_file="./mm10.genome.size", epi_track_files=None,
    tasks=["TARGET"],upsample=False)

    model_predictions=encoder_loaded.predict_generator(test_gen,workers=4,use_multiprocessing=False,verbose=1)
    
    print(model_predictions.shape)

    X_embedded = TSNE(n_components=2).fit_transform(model_predictions)

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1]   )
    plt.colorbar()
    plt.show()

    #model_predictions_bool = model_predictions > 0.5

    #test_db_observed = get_labels_from_target_files("test.bed",["TARGET"])

    #print(ClassificationResult(test_db_observed,model_predictions_bool))





if __name__ == "__main__":
    #initialize_model()

    #example_generator()
    prediction_and_evaluation()

    #print( get_labels_from_target_files("test.bed",["TARGET"]) )