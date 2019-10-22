# this script generate co-binding data. We only consider DNA sequence at this time.

import numpy as np
from random import choice, seed
from helper_funcs import one_hot_encode, randomize
import re

import matplotlib.pyplot as plt


def make_PWM_from_jaspar(jaspar_file = None):

    '''
    read in position weight matrix in jaspar format
    input: file path and name
    output: PWM as a numpy matrix
    '''
    with open(jaspar_file) as f:
        for line in f:
            if line[0] == "A":
                A_list = list(map( float,  line.strip()[7:-2].split() ) )
            if line[0] == "C":
                C_list = list(map( float,  line.strip()[7:-2].split() ) )
            if line[0] == "G":
                G_list = list(map( float,  line.strip()[7:-2].split() ) )
            if line[0] == "T":
                T_list = list(map( float,  line.strip()[7:-2].split() ) )
        
        PWM = np.matrix( [A_list, C_list, G_list, T_list] )

        return  (PWM/PWM.sum(axis=0)  )


def make_motif(jaspar_file):
    '''
    input: PWM matrix (PWM)
    return: a motifi sequence as a python string
    '''
    PWM = make_PWM_from_jaspar(jaspar_file)

    (_,motif_wide) = PWM.shape
    
    
    s_numbers = np.random.uniform(0, 1, motif_wide)

    motif = ""

    for n in range(0,motif_wide):
        if s_numbers[n] < PWM.item((0,n)):
            motif += "A"
        elif s_numbers[n] < PWM.item((0,n)) + PWM.item((1,n))  :
            motif += "C"
        elif s_numbers[n] < PWM.item((0,n)) + PWM.item((1,n)) +  PWM.item((2,n)):
            motif += "G"
        else:
            motif += "T"
    
    return(motif)



def generate_random_DNA_seq(seq_length):
    DNA = "" 
    for i in range(0, seq_length):
        DNA += choice("ACGT")
    
    return(DNA)


def make_a_positive_DNA_seq_single_motif(seq_length, center_pos, PWM_file = None):

    motif_width = make_PWM_from_jaspar(PWM_file).shape[1]

    background_seq = generate_random_DNA_seq(seq_length)

    motif_start_pos = int(seq_length/2) - int(center_pos/2) - motif_width

    motif_end_pos = int(seq_length/2) + int(center_pos/2)
    
    motif_pos = int(np.random.uniform(low=motif_start_pos, high=motif_end_pos))

    positive_seq = background_seq[0:motif_pos] + make_motif(PWM_file) + background_seq[(motif_pos + motif_width):]

    return(positive_seq)


def make_a_positive_DNA_seq_co_binding(seq_length, center_pos, PWM_file_1 = None, PWM_file_2 = None, interspace=None ):

    motif_width_1 = make_PWM_from_jaspar(PWM_file_1).shape[1]

    motif_width_2 = make_PWM_from_jaspar(PWM_file_2).shape[1]

    background_seq = generate_random_DNA_seq(seq_length)

    motif_start_pos = int(seq_length/2) - int(center_pos/2) - (motif_width_1 + motif_width_2 + interspace)

    motif_end_pos = int(seq_length/2) + int(center_pos/2)
    
    motif_pos = int(np.random.uniform(low=motif_start_pos, high=motif_end_pos))

    # positive_seq = background_seq[0:motif_pos] + make_motif(PWM_file_1) + generate_random_DNA_seq(interspace)  + make_motif(PWM_file_2) + background_seq[(motif_pos + motif_width):]

    # positive_seq = background_seq[0:motif_pos] + make_motif(PWM_file_1) + generate_random_DNA_seq(interspace)  + make_motif(PWM_file_2) + background_seq[(motif_pos + motif_width_1 + motif_width_2 + interspace):]

    positive_seq = background_seq[0:motif_pos] + make_motif(PWM_file_1) + generate_random_DNA_seq(interspace)  + make_motif(PWM_file_2) + background_seq[(motif_pos + motif_width_1 + motif_width_2 + interspace):]

    return(positive_seq)



def get_simulated_dataset(parameters, train_size, test_size ):

    motif1 = generate_random_DNA_seq(10)

    motif2 = generate_random_DNA_seq(10)

    motif3 = generate_random_DNA_seq(10)

    #seqs_with_motif_1 = [ make_a_positive_DNA_seq_single_motif(seq_length = parameters["seq_length"], center_pos = parameters["center_pos"], PWM_file= parameters["PWM_file_1"] )  for i in range(train_size)  ]
    
    seqs_with_motif_1 = [  generate_random_DNA_seq(507) + motif1 + generate_random_DNA_seq(507)  for i in range(train_size)  ]

    #seqs_with_motif_2 = [ make_a_positive_DNA_seq_single_motif(seq_length = parameters["seq_length"], center_pos = parameters["center_pos"], PWM_file= parameters["PWM_file_2"] )  for i in range(train_size)  ]

    seqs_with_motif_2 = [  generate_random_DNA_seq(507) + motif2 + generate_random_DNA_seq(507)  for i in range(train_size)  ]

    #seqs_with_motif_1_and_2 = [  make_a_positive_DNA_seq_co_binding(seq_length = parameters["seq_length"], center_pos = parameters["center_pos"], PWM_file_1=parameters["PWM_file_1"], PWM_file_2=parameters["PWM_file_2"] , interspace=parameters["interspace"])  for i in range(train_size)  ]

    seqs_with_motif_1_and_2 = [  generate_random_DNA_seq(496) + motif1 +  generate_random_DNA_seq(12) + motif2 + generate_random_DNA_seq(496)  for i in range(train_size)  ]

    train_X = one_hot_encode(seqs_with_motif_1 + seqs_with_motif_2 + seqs_with_motif_1_and_2)

    train_Y = np.repeat( [0,1,2], repeats = [  train_size, train_size, train_size  ] ).reshape((train_size*3, 1))
    
    ##################################

    #seqs_with_motif_1 = [ make_a_positive_DNA_seq_single_motif(seq_length = parameters["seq_length"], center_pos = parameters["center_pos"], PWM_file= parameters["PWM_file_1"] )  for i in range(test_size)  ]
    
    seqs_with_motif_1 = [  generate_random_DNA_seq(507) + motif1 + generate_random_DNA_seq(507)  for i in range(test_size)  ]

    #seqs_with_motif_2 = [ make_a_positive_DNA_seq_single_motif(seq_length = parameters["seq_length"], center_pos = parameters["center_pos"], PWM_file= parameters["PWM_file_2"] )  for i in range(test_size)  ]

    seqs_with_motif_2 = [  generate_random_DNA_seq(507) + motif2 + generate_random_DNA_seq(507)  for i in range(test_size)  ]

    #seqs_with_motif_1_and_2 = [  make_a_positive_DNA_seq_co_binding(seq_length = parameters["seq_length"], center_pos = parameters["center_pos"], PWM_file_1=parameters["PWM_file_1"], PWM_file_2=parameters["PWM_file_2"] , interspace=parameters["interspace"])  for i in range(test_size)  ]

    seqs_with_motif_1_and_2 = [  generate_random_DNA_seq(496) + motif1 +  generate_random_DNA_seq(12) + motif2 + generate_random_DNA_seq(496)   for i in range(test_size)  ]

    test_X = one_hot_encode(seqs_with_motif_1 + seqs_with_motif_2 + seqs_with_motif_1_and_2)

    test_Y = np.repeat( [0,1,2], repeats = [  test_size, test_size, test_size  ] ).reshape((test_size*3, 1))

    ##################################

    train_X_shuffled, train_Y_shuffled = randomize(train_X, train_Y)

    test_X_shuffled, test_Y_shuffled = randomize(test_X, test_Y)

    return([train_X_shuffled, train_Y_shuffled, test_X_shuffled , test_Y_shuffled])

if __name__ == "__main__":
    seed(0)
    np.random.seed(0)

    #print(make_PWM_from_jaspar("/home/qan/Desktop/DeepEpitif/DeepMetif/JASPAR2018_CORE_vertebrates_non-redundant_pfms_jaspar/MA0835.1.jaspar").shape[1])


    #print(make_a_positive_DNA_seq_co_binding(seq_length=100, center_pos = 50, motif_width_1=10, motif_width_2=10, interspace=10))


