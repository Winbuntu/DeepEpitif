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




def make_meth_for_motif(metif_level):
    return( np.array(metif_level))


def generate_random_DNA_seq(seq_length):
    DNA = "" 
    for i in range(0, seq_length):
        DNA += choice("ACGT")
    
    return(DNA)


def make_meth_for_DNA_seqeunce(seq):

    #return(np.repeat(1.0, len(seq))) # this is for debug

    metif_for_DNA_seq = np.repeat(0.0, len(seq))

    CpG_pos = np.asarray([m.start() for m in re.finditer('CG', seq)])

    num_CG = len(CpG_pos)
    
    if num_CG==0:
        return(metif_for_DNA_seq)

    beta_rv = np.random.beta(a=0.1,b=0.1,size = num_CG)
    
    beta_rv_mod = np.where(beta_rv > -0.1, beta_rv, 0)

    np.put(metif_for_DNA_seq, CpG_pos, beta_rv_mod)

    return(metif_for_DNA_seq)


def make_a_positive_DNA_seq(seq_length, center_pos, motif_width,PWM_file = None, low_meth = None, high_meth = None):
    background_seq = generate_random_DNA_seq(seq_length)
    motif_start_pos = int(seq_length/2) - int(center_pos/2) - motif_width
    motif_end_pos = int(seq_length/2) + int(center_pos/2)
    
    #print(motif_start_pos, motif_end_pos)
    motif_pos = int(np.random.uniform(low=motif_start_pos, high=motif_end_pos))

    positive_seq = background_seq[0:motif_pos] + make_motif(PWM_file) + background_seq[(motif_pos + motif_width):]

    return([positive_seq, motif_pos])


def make_negative_examples(seq_length, n_seq):
    neg_seqs = [ generate_random_DNA_seq(seq_length) for i in range(n_seq)]
    
    neg_seq_meth = np.zeros((n_seq, 1, seq_length, 1))

    for i in range(n_seq):
        neg_seq_meth[i,:,:,:] = make_meth_for_DNA_seqeunce(neg_seqs[i]).reshape((1,seq_length,1))

    return([neg_seqs, neg_seq_meth])



def make_positive_examples(seq_length, n_seq, center_pos, motif_width,PWM_file = None, metif_level = None):
    positive_instances = [make_a_positive_DNA_seq(seq_length = seq_length, center_pos = center_pos, motif_width = motif_width, PWM_file = PWM_file) for i in range(n_seq)]

    positive_seqs = [i[0] for i in positive_instances]
    motif_positions = [i[1] for i in positive_instances]
    
    positive_seq_meth = np.zeros((n_seq, 1, seq_length, 1))

    for i in range(n_seq):
        meth_for_this_seq = make_meth_for_DNA_seqeunce(positive_seqs[i])
        #print(make_meth_for_motif)
        np.put(meth_for_this_seq, list(range(motif_positions[i], (motif_positions[i] + motif_width)  ) ), make_meth_for_motif(metif_level=metif_level) )

        positive_seq_meth[i,:,:,:] = meth_for_this_seq.reshape((1,seq_length,1))

    
    return(positive_seqs, positive_seq_meth)



def get_simulated_dataset(parameters, train_size, valid_size , test_size ):
    
    neg_seqs, neg_meth = make_negative_examples(seq_length = parameters["seq_length"], n_seq = int(train_size/2) )
    pos_seqs, pos_meth = make_positive_examples(seq_length = parameters["seq_length"], n_seq = int(train_size/2), center_pos = parameters["center_pos"], motif_width = parameters["motif_width"], PWM_file=parameters["PWM_file"],metif_level=parameters["metif_level"])
    #print(pos_seqs)
    #print(neg_seqs)
    meths = np.concatenate((neg_meth, pos_meth), axis=0)

    train_X = np.concatenate( (one_hot_encode(neg_seqs + pos_seqs).astype("float"), meths) , axis = 3)
    #train_X = one_hot_encode(neg_seqs + pos_seqs).astype("float")
    
    train_Y = np.repeat([0,1], repeats = [int(train_size/2), int(train_size/2)] ).reshape((train_size, 1)).astype("bool")
    #print(train_Y)
###
    neg_seqs, neg_meth = make_negative_examples(seq_length = parameters["seq_length"], n_seq = int(valid_size/2) )
    pos_seqs, pos_meth = make_positive_examples(seq_length = parameters["seq_length"], n_seq = int(valid_size/2), center_pos = parameters["center_pos"], motif_width = parameters["motif_width"], PWM_file=parameters["PWM_file"],metif_level=parameters["metif_level"])

    meths = np.concatenate((neg_meth, pos_meth), axis=0)

    valid_X = np.concatenate( (one_hot_encode(neg_seqs + pos_seqs).astype("float"), meths) , axis = 3)
    #valid_X = one_hot_encode(neg_seqs + pos_seqs).astype("float")


    valid_Y = np.repeat([0,1], repeats = [int(valid_size/2), int(valid_size/2)] ).reshape((valid_size, 1)).astype("bool")
    #print(valid_Y.shape)

###
    neg_seqs, neg_meth = make_negative_examples(seq_length = parameters["seq_length"], n_seq = int(test_size/2) )
    pos_seqs, pos_meth = make_positive_examples(seq_length = parameters["seq_length"], n_seq = int(test_size/2), center_pos = parameters["center_pos"], motif_width = parameters["motif_width"], PWM_file=parameters["PWM_file"],metif_level=parameters["metif_level"])

    meths = np.concatenate((neg_meth, pos_meth), axis=0)

    test_X = np.concatenate( (one_hot_encode(neg_seqs + pos_seqs).astype("float"), meths) , axis = 3)
    #test_X = one_hot_encode(neg_seqs + pos_seqs).astype("float")

    test_Y = np.repeat([0,1], repeats = [int(test_size/2), int(test_size/2)] ).reshape((test_size, 1)).astype("bool")


    ####

    train_X_shuffled, train_Y_shuffled = randomize(train_X, train_Y)
    valid_X_shuffled, valid_Y_shuffled = randomize(valid_X, valid_Y)
    test_X_shuffled, test_Y_shuffled = randomize(test_X, test_Y)

    return([train_X_shuffled, train_Y_shuffled, valid_X_shuffled, valid_Y_shuffled, test_X_shuffled , test_Y_shuffled])

if __name__ == "__main__":
    seed(0)
    np.random.seed(0)

    #print(make_PWM_from_jaspar("/Users/anqin/Qin_BOXSYNC/Projects/DeepMetif/JASPAR2018_CORE_vertebrates_non-redundant_pfms_jaspar/MA0835.1.jaspar").shape)



    #print(generate_random_DNA_seq(100))
    
    #print(make_a_positive_DNA_seq(100,20,10))

    #print([make_a_positive_DNA_seq(100,20,10) for i in range(10)])
    
    #make_negative_examples(10,1):

    
    #print(make_positive_examples(seq_length = 100, n_seq = 2, center_pos = 20, motif_width = 10))

    simutation_parameters = {
        "PWM_file": "/Users/anqin/Qin_BOXSYNC/Projects/DeepMetif/JASPAR2018_CORE_vertebrates_non-redundant_pfms_jaspar/MA0835.1.jaspar",
        "seq_length":100,
        "center_pos": 20,
        "motif_width":14,
        "metif_level" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    }
    [train_X, train_Y, valid_X, valid_Y, test_X , test_Y] = get_simulated_dataset(parameters = simutation_parameters, train_size = 2, valid_size = 2000, test_size = 10)
    
    #np.savetxt("t1.txt", valid_X)
    #np.savetxt("t2.txt", valid_Y)

    #print(valid_X[71])
    #print(valid_Y[71])
    #print(np.sum(valid_Y))
    #print(train_Y[11, :])


    #print(make_meth_for_DNA_seqeunce("CGCGCGCGCGCGCGCG"))

    #for i in range(1000):
    #    print(make_motif("/Users/anqin/Qin_BOXSYNC/Projects/DeepMetif/JASPAR2018_CORE_vertebrates_non-redundant_pfms_jaspar/MA0835.1.jaspar")) 