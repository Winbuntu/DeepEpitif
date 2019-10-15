import os

def bed_peaks_to_input_regions(bed_file, region_length):
    bedout = open(bed_file+"target_file.txt","w")

    with open(bed_file) as f:
        for line in f:
            elements = line.strip().split("\t")
            chr_name = elements[0]
            start = int(elements[1])
            end = int(elements[2])

            center = (start+end)/2

            #print(center)

            new_start = int(center - region_length/2)
            new_end = int(center + region_length/2)

            bedout.write( "\t".join([chr_name, str(new_start), str(new_end)]) + "\n" )
    
    bedout.close()

    os.system("""cat GSM1865013_ATAC-seq_WT_cones_MACS_peaks_rep1.txttarget_file.txt | grep -v "^chrM" > GSM1865013_ATAC-seq_WT_cones_MACS_peaks_rep1.txttarget_file_clean.txt""")

    os.system("bedtools makewindows -g ../GSM1865005_allC.MethylC-seq_WT_rods_rep1.tsv/mm10.genome.size -w 1500 > mm10_1.5kb_bins.bed")
            
    os.system("""bedtools intersect -v -a mm10_1.5kb_bins.bed -b GSM1865013_ATAC-seq_WT_cones_MACS_peaks_rep1.txttarget_file_clean.txt > target_null.bed""")
    
    os.system("""cat target_null.bed | awk -v OFS="\t" '{print $1,$2,$3,0}' > null_target_with_labels.bed""")

    os.system("""cat GSM1865013_ATAC-seq_WT_cones_MACS_peaks_rep1.txttarget_file_clean.txt | awk -v OFS="\t" '{print $1,$2,$3,1}' > Ture_target_with_labels.bed""")

    os.system("""cat Ture_target_with_labels.bed null_target_with_labels.bed | bedtools sort -i stdin > regions_for_learning.bed""")

    os.system("""cat header.txt regions_for_learning.bed > regions_for_learning_with_head.bed""")


    
    

if __name__ == "__main__":
    bed_peaks_to_input_regions("GSM1865013_ATAC-seq_WT_cones_MACS_peaks_rep1.txt", 1500)

        
