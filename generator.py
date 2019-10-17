# generate dataset from real data for trainning 

# here we modify the DataGenerator function, making it can output sample weights


# load entire epigenetic tracks into memory to speed up. use generatr to generate samples when trainning on the fly. 
# open_data_file to load bed files. this is bed file with target classification. 
# XXX to generate epigenetic matrix and keep it in memory. use it to generate epigenetic features. 


from tensorflow.python.keras.utils import Sequence #############
import pandas as pd
import numpy as np
import random
import math 
import pysam
import threading 

from scipy.sparse import csr_matrix

import pyBigWig

from helper_funcs import ltrdict


#from matplotlib import pyplot as plt 

def get_epimatrix_from_bw(bw_list, chr_name, start, end):
    epimatrix_piece = np.zeros(( end-start ,len(bw_list)))

    for i in range(len(bw_list)):
        bw = bw_list[i]
        epimatrix_piece[:,i] = bw.values(chr_name,start,end)
        epimatrix_piece[np.isnan(epimatrix_piece)] = 0

    return(epimatrix_piece)



def separate_dataset(csv_name, chr_list, dataset_name):
    '''
    This function takes csv file name as input, write a hdf5 file
    containing data for selected chromosomes.
    '''
    import pandas as pd
    aaa = pd.read_csv(csv_name,sep = "\t").set_index(['CHR','START','END'])
    selected_df = aaa.loc[chr_list,:]
    selected_df.to_csv(dataset_name,sep = "\t")

genome_dict = {}


def open_epidata_bigwig(genome_size_file,many_data_paths=None):
    
    chr_lengths = []
    chr_names = []

    with open(genome_size_file) as g:
        for line in g:
            chr_names.append(line.strip().split("\t")[0])
            chr_lengths.append( int(line.strip().split("\t")[1] ))
    

    in_matrix_pos = np.cumsum(chr_lengths) - chr_lengths

    # fill the genome_dict
    for i in range(len(chr_names)):
        genome_dict[chr_names[i]] = in_matrix_pos[i]
    
    # create epigenetic track matrix and fill in the data
    #print(sum(chr_lengths))
    epimatrix = np.zeros( (  sum(chr_lengths) , len(many_data_paths) ) )

    for i in range(len(many_data_paths)):

        bw = pyBigWig.open(many_data_paths[i])

        for c in range(len(chr_names)):
            print(chr_names[c])
            
            epimatrix[  np.arange(start=genome_dict[chr_names[c]], stop=genome_dict[chr_names[c]] + chr_lengths[c]  ) ,i]  =  bw.values(chr_names[c], 0, chr_lengths[c], numpy=True)
            #print(bw.values(chr_names[c], 0, chr_lengths[c], numpy=True).shape)
            #print(np.arange(start=genome_dict[chr_names[c]], stop=genome_dict[chr_names[c]] + chr_lengths[c]  ).shape)

    return(epimatrix)

def epidata_continuous_to_binary():
    pass

def epidata_normalize():
    pass


def open_epigata_bed_to_binary(genome_size_file,many_data_paths=None):
    
    chr_lengths = []
    chr_names = []

    with open(genome_size_file) as g:
        for line in g:
            chr_names.append(line.strip().split("\t")[0])
            chr_lengths.append( int(line.strip().split("\t")[1] ))
    

    in_matrix_pos = np.cumsum(chr_lengths) - chr_lengths

    # fill the genome_dict
    for i in range(len(chr_names)):
        genome_dict[chr_names[i]] = in_matrix_pos[i]
    
    # create epigenetic track matrix and fill in the data
    #print(sum(chr_lengths))
    epimatrix = np.zeros( (  sum(chr_lengths) , len(many_data_paths) ) )

    for i in range(len(many_data_paths)):
        with open(many_data_paths[i]) as f:
            for line in f:
                char_name,start,end = line.strip().split("\t")[0], int(line.strip().split("\t")[1]), int(line.strip().split("\t")[2])

                epimatrix[ np.arange( start= genome_dict[char_name] + start, stop =  genome_dict[char_name] + end  ) ,i] = 1

    
    return(epimatrix)


def open_epidata_file(genome_size_file,many_data_paths=None):
    # supports bed file at this point. make a big pandas dataframe, with 3 indexes and multiple columns marks epigenetic data
    # mysql --user=genome --host=genome-mysql.cse.ucsc.edu -A -e "select chrom, size from mm10.chromInfo" | grep -v "chrom" >  mm10.genome.size
    #genome_size = pd.read_csv(genome_size_file, sep = "\t", header = None)
    
    # create data frame
    

    #number_of_epitracks = len(many_data_paths)

    chr_lengths = []
    chr_names = []

    with open(genome_size_file) as g:
        for line in g:
            chr_names.append(line.strip().split("\t")[0])
            chr_lengths.append( int(line.strip().split("\t")[1] ))
    

    in_matrix_pos = np.cumsum(chr_lengths) - chr_lengths

    # fill the genome_dict
    for i in range(len(chr_names)):
        genome_dict[chr_names[i]] = in_matrix_pos[i]
    
    # create epigenetic track matrix and fill in the data
    #print(sum(chr_lengths))
    print(sum(chr_lengths))
    
    #epimatrix = np.zeros( (  sum(chr_lengths) , len(many_data_paths) ), dtype="float32" )
    
    epimatrix = csr_matrix( (  sum(chr_lengths) , len(many_data_paths) ) , dtype=np.float)
    # compute average CpG methylation 

    # embed in average CpG methylation for all CpG sites

    # embed in observed CpG methylation

    # this version is for CpG methylation

    position_list_to_put = []
    value_list_to_put = []

    #print(len(position_list_to_put))
    
    #print(sum(value_list_to_put))

    for i in range(len(many_data_paths)):
        with open(many_data_paths[i]) as f:
            for line in f:
                chr_name,chr_site,strand,trinucl,mCcount,total_count,mCstatus = line.strip().split("\t")
                chr_name = "chr" + chr_name
                chr_site = int(chr_site)
                mCstatus = int(mCstatus)

                position_list_to_put.append(  genome_dict[chr_name] + chr_site  )

                value_list_to_put.append( mCstatus )

        epimatrix[position_list_to_put,i] = value_list_to_put
        
        #print(len(position_list_to_put))
        #print(sum(value_list_to_put))
        #print(len(value_list_to_put))
    return epimatrix


def open_data_file(data_path,tasks,num_to_read=None):
    if data_path.endswith('.hdf5'):
        if (tasks is None) and (num_to_read is not None):
            data=pd.read_hdf(data_path,start=0,stop=num_to_read)
        elif (tasks is not None) and (num_to_read is None):
            data=pd.read_hdf(data_path,columns=tasks)
        elif (tasks is None) and (num_to_read is None):
            data=pd.read_hdf(data_path)
        else: 
            data=pd.read_hdf(data_path,columns=tasks,start=0,stop=num_to_read)
    else:
        #treat as bed file 
        if (tasks is None) and (num_to_read is not None):
            data=pd.read_csv(data_path,header=0,sep='\t',index_col=[0,1,2],start=0,stop=num_to_read)
        elif (tasks is None) and (num_to_read is None):
            data=pd.read_csv(data_path,header=0,sep='\t',index_col=[0,1,2])
        else:
            data=pd.read_csv(data_path,header=0,sep='\t',nrows=1)
            chrom_col=data.columns[0]
            start_col=data.columns[1]
            end_col=data.columns[2]
            if num_to_read is None:
                data=pd.read_csv(data_path,header=0,sep='\t',usecols=[chrom_col,start_col,end_col]+tasks,index_col=[0,1,2])
            else:
                data=pd.read_csv(data_path,header=0,sep='\t',usecols=[chrom_col,start_col,end_col]+tasks,index_col=[0,1,2],start=0,stop=num_to_read)
    return data 


class DataGenerator(Sequence):
    def __init__(self,data_path,ref_fasta,genome_size_file = None, epi_track_files = None,batch_size=128,tasks=None,upsample=True,upsample_ratio=0.1,upsample_type=1,num_to_read=None):
        self.lock = threading.Lock()        
        self.batch_size=batch_size


        #open the reference file
        self.ref_fasta=ref_fasta
        self.data=open_data_file(data_path,tasks,num_to_read)
        
        #self.epidata = open_epidata_file(genome_size_file, epi_track_files) # 
        self.epi_track_files =  epi_track_files
        
        if self.epi_track_files is not None:
            self.bw_opened_list = []
            for i in range(len( self.epi_track_files )):
                self.bw_opened_list.append(  pyBigWig.open( self.epi_track_files[i] )  )

        #bw_handles = [pyBigWig.open(epi_track_files[i]) for i in range(len(epi_track_files))]

        self.indices=np.arange(self.data.shape[0])
        num_indices=self.indices.shape[0]
        #self.add_revcomp=add_revcomp
        
        #set variables needed for upsampling the positives
        self.upsample=upsample
        if self.upsample==True:
            self.upsample_ratio=upsample_ratio
            self.upsample_type=upsample_type
            self.ones = self.data.loc[(self.data == self.upsample_type).any(axis=1)]
            self.zeros = self.data.loc[(self.data != self.upsample_type).all(axis=1)]
            self.pos_batch_size = int(self.batch_size * self.upsample_ratio)
            self.neg_batch_size = self.batch_size - self.pos_batch_size
            self.pos_indices=np.arange(self.ones.shape[0])
            self.neg_indices=np.arange(self.zeros.shape[0])
            

            #wrap the positive and negative indices to reach size of self.indices
            num_pos_wraps=math.ceil(num_indices/self.pos_indices.shape[0])
            num_neg_wraps=math.ceil(num_indices/self.neg_indices.shape[0])
            self.pos_indices=np.repeat(self.pos_indices,num_pos_wraps)[0:num_indices]
            np.random.shuffle(self.pos_indices)
            self.neg_indices=np.repeat(self.neg_indices,num_neg_wraps)[0:num_indices]
            np.random.shuffle(self.neg_indices)
            
    def __len__(self):
        return math.ceil(self.data.shape[0]/self.batch_size)

    def __getitem__(self,idx):
        with self.lock:
            self.ref=pysam.FastaFile(self.ref_fasta)
            
            #if self.shuffled_ref_negatives==True:
            #    return self.get_shuffled_ref_negatives_batch(idx)
            
            if self.upsample==True:
                return self.get_upsampled_positives_batch(idx)
            else:
                return self.get_basic_batch(idx) 



    def get_upsampled_positives_batch(self,idx):
        #get seq positions
        pos_inds=self.pos_indices[idx*self.pos_batch_size:(idx+1)*self.pos_batch_size]
        pos_bed_entries=self.ones.index[pos_inds]
        neg_inds=self.neg_indices[idx*self.neg_batch_size:(idx+1)*self.neg_batch_size]
        neg_bed_entries=self.zeros.index[neg_inds]
    

        #get sequences
        pos_seqs=[self.ref.fetch(i[0],i[1],i[2]) for i in pos_bed_entries]
        neg_seqs=[self.ref.fetch(i[0],i[1],i[2]) for i in neg_bed_entries]
        seqs=pos_seqs+neg_seqs 
        


        #one-hot-encode the fasta sequences 
        seqs=np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
        x_batch=np.expand_dims(seqs,1)
        
        #extract the positive and negative labels at the current batch of indices
        y_batch_pos=self.ones.iloc[pos_inds]
        y_batch_neg=self.zeros.iloc[neg_inds]
        y_batch=np.concatenate((y_batch_pos,y_batch_neg),axis=0)
        #add in the labels for the reverse complement sequences, if used 

        if self.epi_track_files is None:
            return(x_batch,y_batch)
        
        x_epidata_batch_pos =  np.array([  [get_epimatrix_from_bw( self.bw_opened_list  , chr_name = i[0], start = i[1], end = i[2])] for i in pos_bed_entries ])
        x_epidata_batch_neg =  np.array([  [get_epimatrix_from_bw( self.bw_opened_list  , chr_name = i[0], start = i[1], end = i[2])] for i in neg_bed_entries ])
        x_epidata_batch = np.concatenate( (x_epidata_batch_pos,x_epidata_batch_neg), axis=0 )
        x_batch_full = np.concatenate((x_batch,x_epidata_batch),axis=3)

        return (x_batch_full,y_batch)            
    
    def get_basic_batch(self,idx):
        #get seq positions
        inds=self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        bed_entries=self.data.index[inds]
        #get sequences
        seqs=[self.ref.fetch(i[0],i[1],i[2]) for i in bed_entries]
        

        
        #one-hot-encode the fasta sequences 
        seqs=np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
        x_batch=np.expand_dims(seqs,1)
        #extract the labels at the current batch of indices 
        y_batch=np.asarray(self.data.iloc[inds])

        if self.epi_track_files is None:
            return(x_batch,y_batch)

        # get epidata pieces and combine it with seqeunce one-hot matrix
        x_epidata_batch =  np.array([  [get_epimatrix_from_bw( self.bw_opened_list  , chr_name = i[0], start = i[1], end = i[2])] for i in bed_entries])
        x_batch_full = np.concatenate((x_batch,x_epidata_batch),axis=3)
        
        return (x_batch_full,y_batch)    
    
    def on_epoch_end(self):
        #if upsampling is being used, shuffle the positive and negative indices 
        if self.upsample==True:
            np.random.shuffle(self.pos_indices)
            np.random.shuffle(self.neg_indices)
        else:
            np.random.shuffle(self.indices)



if __name__ == "__main__":

    print( np.array([ [get_epimatrix_from_bw(many_data_paths = ["MethylC-seq_WT_cones_rep1_CpG.clean.plus.sorted.bw"], 
    chr_name = "chr1", start = 3000827, end = 3000840) ] , 
    [get_epimatrix_from_bw(many_data_paths = ["MethylC-seq_WT_cones_rep1_CpG.clean.plus.sorted.bw"], 
    chr_name = "chr1", start = 3000827, end = 3000840) ]]).shape)

    #pass
    #pp =open_epidata_file("mm10.genome.size", many_data_paths=["../GSM1865005_allC.MethylC-seq_WT_rods_rep1.tsv/AM_R2_allChr_CpG_noL.txt","../GSM1865005_allC.MethylC-seq_WT_rods_rep1.tsv/AM_R2_allChr_CpG_noL.txt" ])
    #print(pp.sum(axis=0))
    #print( np.expand_dims(np.array( [ pp[1:10,:] , pp[20:30,:] ] )  , axis = 1 ).shape)
    #print(genome_dict)

    #print(  np.array([  [  pp[1:1500,:]   ]   , [  pp[1:1500,:]  ]  ]).shape )

    #print(open_epigata_bed_to_binary("mm10.genome.size", many_data_paths=["/Users/anqin/Desktop/DeepEpitif/GSE72550/GSE72550_MethylC-seq_rd7_rods_UMRs_LMRs.txt" ]).shape)

    #a = open_epidata_bigwig("mm10.genome.size.clean", ["/Users/anqin/Desktop/DeepEpitif/GSE72550/GSE72550_RAW/GSM1865011_ATAC-seq_WT_rods_rep1_scaled10M.bw"])
    #print(a.max())
    #plt.hist( np.log10(a[:,0]+1), bins = 100) 
    #plt.title("histogram") 
    #plt.show()