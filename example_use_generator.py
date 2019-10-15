from keras.models import Sequential
from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adadelta, SGD, RMSprop
import keras.losses
from keras.constraints import maxnorm
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping, History
from keras import backend as K 
from keras.models import model_from_json


K.set_image_data_format('channels_last')

from random import choice, seed
import numpy as np

from simulations import get_simulated_dataset

from helper_funcs import MetricsCallback

from interpret import plot_CNN_filters


from generator import DataGenerator






def example_generator():
    
    train_gen = DataGenerator(data_path="/Users/anqin/Desktop/GSM1865005_allC.MethylC-seq_WT_rods_rep1.tsv/data.sort.clean.bed", 
    ref_fasta = "/Users/anqin/Desktop/GSM1865005_allC.MethylC-seq_WT_rods_rep1.tsv/GRCm38.primary_assembly.genome.fa.gz",
    genome_size_file="./mm10.genome.size", epi_track_files=["/Users/anqin/Desktop/GSM1865005_allC.MethylC-seq_WT_rods_rep1.tsv/AM_R2_allChr_CpG_noL.txt"],
    tasks=["TARGET"],upsample=False)
    
    valid_gen = DataGenerator(data_path="/Users/anqin/Desktop/GSM1865005_allC.MethylC-seq_WT_rods_rep1.tsv/data_valid_sort.clean.bed", 
    ref_fasta = "/Users/anqin/Desktop/GSM1865005_allC.MethylC-seq_WT_rods_rep1.tsv/GRCm38.primary_assembly.genome.fa.gz",
    genome_size_file="./mm10.genome.size", epi_track_files=["/Users/anqin/Desktop/GSM1865005_allC.MethylC-seq_WT_rods_rep1.tsv/AM_R2_allChr_CpG_noL.txt"],
    tasks=["TARGET"],upsample=False)

    one_filter_keras_model=Sequential() 
    one_filter_keras_model.add(Conv2D(filters=5,kernel_size=(1,15),padding="same",input_shape=  (1,1000,5)  ))
    one_filter_keras_model.add(BatchNormalization(axis=-1))
    one_filter_keras_model.add(Activation('relu'))
    one_filter_keras_model.add(MaxPooling2D(pool_size=(1,35)))
    one_filter_keras_model.add(Flatten())
    one_filter_keras_model.add(Dense(1))
    one_filter_keras_model.add(Activation("sigmoid"))
    one_filter_keras_model.summary()

    one_filter_keras_model.compile(optimizer='adam',loss='binary_crossentropy')


    #metrics_callback=MetricsCallback(train_data=(train_X,train_Y),
    #                             validation_data=(valid_X,valid_Y))

    
    #print(one_filter_keras_model.get_weights())

    history_regression=one_filter_keras_model.fit_generator(train_gen,
                                                  validation_data=valid_gen,
                                                  steps_per_epoch=500,
                                                  validation_steps=100,
                                                  epochs=150,
                                                  verbose=1,
                                                  use_multiprocessing=False,
                                                  workers=1,
                                                  max_queue_size=50,
                callbacks=[History() ])




if __name__ == "__main__":
    example_generator()