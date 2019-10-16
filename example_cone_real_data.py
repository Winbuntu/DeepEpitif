from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.python.keras.optimizers import Adadelta, SGD, RMSprop
import tensorflow.python.keras.losses
from tensorflow.python.keras.constraints import maxnorm
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.regularizers import l1, l2
from tensorflow.python.keras.callbacks import EarlyStopping, History, ModelCheckpoint
from tensorflow.python.keras import backend as K 
from tensorflow.python.keras.models import model_from_json

import pandas as pd

K.set_image_data_format('channels_last')

from random import choice, seed
import numpy as np

from simulations import get_simulated_dataset

from helper_funcs import *

from interpret import plot_CNN_filters


from generator import DataGenerator,separate_dataset

from runtime_metrics import *



from numpy.random import seed
seed(1234)
from tensorflow import set_random_seed
set_random_seed(1234)

def initialize_model():

    one_filter_keras_model=Sequential() 
    one_filter_keras_model.add(Conv2D(filters=10,kernel_size=(1,15),padding="same",input_shape=  (1,1500,5)  ))
    one_filter_keras_model.add(BatchNormalization(axis=-1))
    one_filter_keras_model.add(Activation('relu'))
    one_filter_keras_model.add(MaxPooling2D(pool_size=(1,35)))
    one_filter_keras_model.add(Flatten())
    one_filter_keras_model.add(Dense(1))
    one_filter_keras_model.add(Activation("sigmoid"))
    one_filter_keras_model.summary()
    one_filter_keras_model.compile(optimizer='adam',loss='binary_crossentropy', metrics=[precision, recall, specificity] )

    return one_filter_keras_model


def example_generator():
    
    separate_dataset("regions_for_learning_with_head.clean.equal_size.bed", ["chr1"], "valid.bed")
    separate_dataset("regions_for_learning_with_head.clean.equal_size.bed", ["chr2","chr19"], "test.bed")
    separate_dataset("regions_for_learning_with_head.clean.equal_size.bed", ["chr3","chr4","chr5","chr6","chr7","chr8",
                                                            "chr9","chr10","chr11","chr12","chr13",
                                                            "chr14","chr15","chr16","chr17",
                                                            "chr18","chr20","chr21","chr22"], "train.bed")
                                                            
    

    train_gen = DataGenerator(data_path="train.bed", 
    ref_fasta = "../GSM1865005_allC.MethylC-seq_WT_rods_rep1.tsv/GRCm38.primary_assembly.genome.fa.gz",
    genome_size_file="./mm10.genome.size", epi_track_files=["MethylC-seq_WT_cones_rep1_CpG.clean.plus.sorted.bw"],
    tasks=["TARGET"],upsample=True,upsample_ratio=0.3)
    
    valid_gen = DataGenerator(data_path="valid.bed", 
    ref_fasta = "../GSM1865005_allC.MethylC-seq_WT_rods_rep1.tsv/GRCm38.primary_assembly.genome.fa.gz",
    genome_size_file="./mm10.genome.size", epi_track_files=["MethylC-seq_WT_cones_rep1_CpG.clean.plus.sorted.bw"],
    tasks=["TARGET"],upsample=True,upsample_ratio=0.3)


    model = initialize_model()


    trainning_history=model.fit_generator(train_gen,
                                                  validation_data=valid_gen,
                                                  steps_per_epoch=500,
                                                  validation_steps=100,
                                                  epochs=20,
                                                  verbose=1,
                                                  use_multiprocessing=False,
                                                  workers=1,
                                                  max_queue_size=50,
                callbacks=[History(), ModelCheckpoint("ATAC_peak_Classification.h5", 
                                           monitor='val_loss', verbose=1, save_best_only=True, mode='min') ])


def prediction_and_evaluation():
    from tensorflow.python.keras.models import load_model

    model = initialize_model()
    model.load_weights("ATAC_peak_Classification.h5")

    #Get predictions on the test set 

    test_gen = DataGenerator(data_path="test.bed", 
    ref_fasta = "../GSM1865005_allC.MethylC-seq_WT_rods_rep1.tsv/GRCm38.primary_assembly.genome.fa.gz",
    genome_size_file="./mm10.genome.size", epi_track_files=["MethylC-seq_WT_cones_rep1_CpG.clean.plus.sorted.bw"],
    tasks=["TARGET"],upsample=False)


    model_predictions=model.predict_generator(test_gen,workers=4,use_multiprocessing=False,verbose=1)

    model_predictions_bool = model_predictions > 0.5

    test_db_observed = get_labels_from_target_files("test.bed",["TARGET"])

    print(ClassificationResult(test_db_observed,model_predictions_bool))





if __name__ == "__main__":
    #example_generator()
    prediction_and_evaluation()

    #print( get_labels_from_target_files("test.bed",["TARGET"]) )