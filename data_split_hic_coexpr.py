import os, time, sys, math,random
import numpy as np
import hickle as hkl
import pandas as pd
#import pickle
import datetime

# prepare in parallel coexpr and hic so that i can discard patch concurrently

#setDir = ""
#setDir = "/media/electron"
#matFile = setDir + "/mnt/etemp/marie/nn_coexpr/Hi-C_MCF7_MCF10A_processed_HiCfiles/Heatmaps/chrxchr/40kb/HiCStein-MCF7-WT__hg19__chr1__C-40000-iced.matrix"
#coexprFile = setDir + "/mnt/etemp/marie/nn_coexpr/INPUT_AGG_40kb/GSE71862_MCF7_MCF10A_RSEM_expectedcounts_chr1_agg.txt"
#assert os.path.exists(matFile)
#hicMat = pd.read_csv(matFile, header=0, index_col=0, sep="\t")
#assert hicMat.shape[0] == hicMat.shape[1]
#hicDT = hicMat.values

# python data_split_hic_coexpr.py 

# python data_split_hic_coexpr.py <HIC_INPUT_DIR> <HIC_FILE_PREFIX> <HIC_FILE_SUFFIX> <COEXPR_INPUT_DIR> <COEXPR_FILE_PREFIX> <COEXPR_FILE_SUFFIX> <OUT_DIR>
# python data_split_hic_coexpr.py Hi-C_MCF7_MCF10A_processed_HiCfiles/Heatmaps/chrxchr/40kb HiCStein-MCF7-WT__hg19_ _C-40000-iced.matrix INPUT_AGG_40kb GSE71862_MCF7_MCF10A_RSEM_expectedcounts _agg.txt INPUT_SPLIT_HiC

startTime = datetime.datetime.now()

setDir=""

if len(sys.argv) == 1:

    HIC_input_dir = "Hi-C_MCF7_MCF10A_processed_HiCfiles/Heatmaps/chrxchr/40kb"
    HIC_file_prefix = "HiCStein-MCF7-WT__hg19_"
    HIC_file_suffix = "_C-40000-iced.matrix"

    COEXPR_input_dir = "INPUT_AGG_40kb"
    COEXPR_file_prefix = "GSE71862_MCF7_MCF10A_RSEM_expectedcounts"
    COEXPR_file_suffix = "agg.txt"

    output_dir = "INPUT_SPLIT_HiC_COEXPR"

else:

    if len(sys.argv) != 8:
        print("ERROR: invalid number of arguments!")
        sys.exit(1)

    HIC_input_dir = sys.argv[1]
    HIC_file_prefix = sys.argv[2]
    HIC_file_suffix = sys.argv[3]
    COEXPR_input_dir =  sys.argv[4]
    COEXPR_file_prefix = sys.argv[5]
    COEXPR_file_suffix = sys.argv[6]
    output_dir = sys.argv[7]


out_dir = os.path.join(output_dir)
os.makedirs(out_dir, exist_ok=True)

logFile = os.path.join(out_dir, "split_hic_coexpr_logFile.txt")
print("... write logs in:\t" + logFile)

if os.path.exists(logFile):
    os.remove(logFile)

if not os.path.exists(HIC_input_dir):
    print("ERROR: Hi-C data path wrong !")
    sys.exit(1)

if not os.path.exists(COEXPR_input_dir):
    print("ERROR: coexpr data path wrong !")
    sys.exit(1)


### HARD-CODED STUFF:
chromo_list = list(range(1,23))   #chr1-chr22
train_chromo_list = list(range(1,18))
test_chromo_list = list(range(18,23))
# for 1-4
chromo_list = list(range(1,5))   #chr1-chr22
train_chromo_list = list(range(1,3))
test_chromo_list = list(range(3,5))



#chromo_list = list(range(1,2))   #chr1-chr22
#train_chromo_list = list(range(1,2))
#test_chromo_list = list(range(1,2))


size_file = setDir+"/mnt/etemp/marie/hicGAN/hicgan_virtual/KARPAS_data/all_chr_length.txt"
assert os.path.exists(size_file)
chrSizeDict = {item.split()[0]:int(item.strip().split()[1]) for item in open(size_file).readlines()}

bin_size = 40000
# do not restrict distances ??? # initially 200; set to work for 10kb data -> 200*10*1000 = 2 MB
maxDistBin = math.inf
# 10 ? initially 40; set to work for 10kb data -> 40*10kb = 400 kb
patchSizeBin = 10

print("!!! HARD-CODED: ")
print("chromo_list:")
print(chromo_list)
print("train_chromo_list:")
print(train_chromo_list)
print("test_chromo_list:")
print(test_chromo_list)
print("bin_size = " + str(bin_size))
print("maxDistBin = " + str(maxDistBin))
print("patchSizeBin = " + str(patchSizeBin))
#################################################################################################################################################################################
################################################################################################################################################################################# def hic_matrix_extraction
################################################################################################################################################################################# 

def hic_matrix_extraction(coexpr_inDir, hic_inDir, coexpr_filePrefix, hic_filePrefix, coexpr_fileSuffix, hic_fileSuffix, chrSizeDict, binSize, idxInBin=True, hicFormat="dekker"):

    # PREPARE THE COEXPR DATA (= target)
    print("*** PREPARING COEXPR DATA")
    coexpr_contacts_dict={}
    for chrom in chromo_list:
        chromo = "chr" + str(chrom)
        chr_size = chrSizeDict[chromo]
        mat_dim = int(math.ceil(chr_size*1.0/binSize))
        print("> START " + chromo)
        print("... size = " + str(chr_size) + " (" + str(mat_dim) + " bins)")
        coexpr_file = '%s/%s_%s_%s'%(coexpr_inDir, coexpr_filePrefix, chromo, coexpr_fileSuffix)
        print(coexpr_file)
        assert os.path.exists(coexpr_file)
        print("... build coexpr matrix from:\t" + coexpr_file)
        coexpr_contact_matrix = np.zeros((mat_dim,mat_dim))
        # UPDATE: base quality threshold on # of NAN
        coexpr_contact_matrix.fill(np.nan)
        for line in open(coexpr_file).readlines():
            idx1, idx2, value = int(line.strip().split('\t')[0]),int(line.strip().split('\t')[1]),float(line.strip().split('\t')[2])        
            #print("idx1 = " + str(idx1) + " - idx2 = " + str(idx2) + " - value = " + str(value))
            if not idxInBin:
                idx1 /= binSize
                idx2 /= binSize
            assert idx1 <= idx2
            assert idx1 <= mat_dim
            assert idx2 <= mat_dim
            coexpr_contact_matrix[idx1][idx2] = value
            
        # end-for iterating over interactions (lines in file)
        coexpr_contact_matrix+= coexpr_contact_matrix.T - np.diag(coexpr_contact_matrix.diagonal())
        coexpr_contacts_dict[chromo] = coexpr_contact_matrix
    # end-for iterating over chromosomes

    # PREPARE THE Hi-C DATA (=predictor)
    print("*** PREPARING Hi-C DATA")
    hic_contacts_dict={}
    for chrom in chromo_list:
        chromo = "chr" + str(chrom)
        chr_size = chrSizeDict[chromo]
        mat_dim = int(math.ceil(chr_size*1.0/binSize))
        print("> START " + chromo)
        print("... size = " + str(chr_size) + " (" + str(mat_dim) + " bins)")
        hic_file = '%s/%s_%s_%s'%(hic_inDir, hic_filePrefix, chromo, hic_fileSuffix)
        print(hic_file)
        assert os.path.exists(hic_file)
        print("... build hi-c matrix from:\t" + hic_file)

        if hicFormat == "dekker":
            hic_contact_dt = pd.read_csv(hic_file, header=0, index_col=0, sep="\t")
            assert hic_contact_dt.shape[0] == hic_contact_dt.shape[1]
            hic_contact_matrix = hic_contact_dt.values
            # replace nan values with 0
            # hic_contact_matrix = np.nan_to_num(hic_contact_matrix, 0)  # modified in place but otherwise printed
            # => UPDATE: base quality threshold on  # of NAN
            assert hic_contact_matrix.shape[0] == hic_contact_matrix.shape[1]
            print("... build from dekker: OK")

            hic_contact_matrix

        elif hicFormat == "agg":
            hic_contact_matrix = np.zeros((mat_dim,mat_dim))
            # UPDATE: base quality threshold on # of NAN
            hic_contact_matrix.fill(np.nan)
            for line in open(hic_file).readlines():
                idx1, idx2, value = int(line.strip().split('\t')[0]),int(line.strip().split('\t')[1]),float(line.strip().split('\t')[2])        
                #print("idx1 = " + str(idx1) + " - idx2 = " + str(idx2) + " - value = " + str(value))
                if not idxInBin:
                    idx1 /= int(binSize)
                    idx2 /= int(binSize)
                assert idx1 <= idx2
                assert idx1 <= mat_dim
                assert idx2 <= mat_dim
                hic_contact_matrix[idx1][idx2] = value
            # end-for iterating over iterations (lines in file)
            hic_contact_matrix+= hic_contact_matrix.T - np.diag(hic_contact_matrix.diagonal())
        else:
            print("ERROR: unimplemented hicFormat")
            sys.exit(1)

        hic_contacts_dict[chromo] = hic_contact_matrix

    # end-for iterating over chromosomes
#    df = open("tmp.pkl", 'wb')
#    pickle.dump(hic_contacts_dict, df)
#    df.close()

    # DO NOT COUNT THE NAN VALUES FOR THE OUTPUT ???
    #nb_coexpr_contacts={item:sum(sum(coexpr_contacts_dict[item])) for item in coexpr_contacts_dict.keys()}
    nb_coexpr_contacts={item:sum(sum(np.nan_to_num(coexpr_contacts_dict[item],0))) for item in coexpr_contacts_dict.keys()}
    #nb_hic_contacts={item:sum(sum(hic_contacts_dict[item])) for item in hic_contacts_dict.keys()}
    nb_hic_contacts={item:sum(sum(np.nan_to_num(hic_contacts_dict[item], 0))) for item in hic_contacts_dict.keys()}
    
    return coexpr_contacts_dict,hic_contacts_dict,nb_coexpr_contacts,nb_hic_contacts

#################################################################################################################################################################################
################################################################################################################################################################################# def crop_hic_matrix_by_chrom
################################################################################################################################################################################# 

def crop_hic_matrix_by_chrom(chrom,norm_type,patch_size,thresh_dist):
    #thred=2M/binSizeolution
    #norm_type=0-->raw count
    #norm_type=1-->log transformation
    #norm_type=2-->scaled to[-1,1]after log transformation, default
    #norm_type=3-->scaled to[0,1]after log transformation
    distance=[]
    crop_mats_coexpr=[]
    crop_mats_hic=[]    
    row_coexpr,col_coexpr = coexpr_contacts_norm_dict[chrom].shape
    row_hic,col_hic = hic_contacts_norm_dict[chrom].shape

    row = min(row_coexpr, row_hic)
    col = min(col_coexpr, col_hic)

    assert row == col
    

    if thresh_dist != math.inf:
        if row <= thresh_dist or col <= thresh_dist:
            print('HiC matrix size wrong!')
            sys.exit(1)

    def quality_control_hic(mat,quality_thresh=0.05):
        # first check number of nan
        if (~np.isnan(mat)).sum() < quality_thresh*mat.shape[0]*mat.shape[1]:
            return False
        # if passed nan thresh, replace nan with 0 before checking number of non zero
        mat = np.nan_to_num(mat, 0)
        if len(mat.nonzero()[0]) < quality_thresh*mat.shape[0]*mat.shape[1]:
            return False
        else:
            return True


    def quality_control_coexpr(mat,quality_thresh=0.05):
        # in coexpr, check only number of nan
        if (~np.isnan(mat)).sum() < quality_thresh*mat.shape[0]*mat.shape[1]:
            return False
        else:
            return True



    tot_idx = 0    
    tot_quality = 0        

    for idx1 in range(0,row-patch_size,patch_size):
        for idx2 in range(0,col-patch_size,patch_size):
            tot_idx +=1 
            if abs(idx1-idx2)<thresh_dist:
                if quality_control_hic(hic_contacts_norm_dict[chrom][idx1:idx1+patch_size,idx2:idx2+patch_size]) and quality_control_coexpr(coexpr_contacts_norm_dict[chrom][idx1:idx1+patch_size,idx2:idx2+patch_size]):
                    tot_quality += 1
                    distance.append([idx1-idx2,chrom])
                    if norm_type==0:
                        hic_contact = hic_contacts_dict[chrom][idx1:idx1+patch_size,idx2:idx2+patch_size]
                        coexpr_contact = coexpr_contacts_dict[chrom][idx1:idx1+patch_size,idx2:idx2+patch_size]
                    elif norm_type==1:
                        hic_contact = hic_contacts_norm_dict[chrom][idx1:idx1+patch_size,idx2:idx2+patch_size]
                        coexpr_contact = coexpr_contacts_norm_dict[chrom][idx1:idx1+patch_size,idx2:idx2+patch_size]
                    elif norm_type==2:
                        hic_contact_norm = hic_contacts_norm_dict[chrom][idx1:idx1+patch_size,idx2:idx2+patch_size]
                        coexpr_contact_norm = coexpr_contacts_norm_dict[chrom][idx1:idx1+patch_size,idx2:idx2+patch_size]
                        hic_contact = hic_contact_norm*2.0/max_hic_contact_norm[chrom]-1
                        coexpr_contact = coexpr_contact_norm*2.0/max_coexpr_contact_norm[chrom]-1
                    elif norm_type==3:
                        hic_contact_norm = hic_contacts_norm_dict[chrom][idx1:idx1+patch_size,idx2:idx2+patch_size]
                        coexpr_contact_norm = coexpr_contacts_norm_dict[chrom][idx1:idx1+patch_size,idx2:idx2+patch_size]
                        hic_contact = hic_contact_norm*1.0/max_hic_contact_norm[chrom]
                        coexpr_contact = coexpr_contact_norm*1.0/max_coexpr_contact_norm[chrom]
                    else:
                        print('Normalization wrong!')
                        sys.exit()
                    
                    # WHAT TO DO WITH THE NAN ??? after the quality check, replace the NaN with 0 - for coexpr ???
                    hic_contact = np.nan_to_num(hic_contact, 0)
                    coexpr_contact = np.nan_to_num(coexpr_contact, 0)

                    crop_mats_hic.append(hic_contact)
                    crop_mats_coexpr.append(coexpr_contact)

    crop_mats_coexpr = np.concatenate([item[np.newaxis,:] for item in crop_mats_coexpr],axis=0)
    crop_mats_hic = np.concatenate([item[np.newaxis,:] for item in crop_mats_hic],axis=0)

    
    mylog = open(logFile,"a+") 
    mylog.write(chrom + " - pass quality control:\t" + str(tot_quality) + "/" + str(tot_idx) + "\n")
    mylog.close() 

    return crop_mats_coexpr,crop_mats_hic,distance

#################################################################################################################################################################################
################################################################################################################################################################################# def data_split
################################################################################################################################################################################# 



def data_split(chromo_list,norm_type, train_data):
    random.seed(100)
    distance_all=[]
    assert len(chromo_list) > 0
    hr_mats,lr_mats=[],[]
    for chrom in chromo_list:
        crop_mats_coexpr,crop_mats_hic,distance = crop_hic_matrix_by_chrom(chrom=chrom, norm_type=norm_type, patch_size=patchSizeBin ,thresh_dist=maxDistBin) # returns coexpr,hic,dist
        distance_all+=distance
        hr_mats.append(crop_mats_coexpr)
        lr_mats.append(crop_mats_hic)
    hr_mats = np.concatenate(hr_mats,axis=0)
    lr_mats = np.concatenate(lr_mats,axis=0)
    hr_mats=hr_mats[:,np.newaxis]
    lr_mats=lr_mats[:,np.newaxis]
    hr_mats=hr_mats.transpose((0,2,3,1))
    lr_mats=lr_mats.transpose((0,2,3,1))

    if train_data:
        hr_train_shuffle_list = list(range(len(hr_mats)))
        lr_train_shuffle_list = list(range(len(lr_mats)))
        hr_mats = hr_mats[hr_train_shuffle_list]
        lr_mats = lr_mats[lr_train_shuffle_list]

    return hr_mats,lr_mats,distance_all



#################################################################################################################################################################################


print("> extract Hi-C data and prepare matrices... ")


coexpr_contacts_dict,hic_contacts_dict,nb_coexpr_contacts,nb_hic_contacts = hic_matrix_extraction(hic_inDir=HIC_input_dir, hic_filePrefix=HIC_file_prefix, hic_fileSuffix=HIC_file_suffix,coexpr_inDir=COEXPR_input_dir, coexpr_filePrefix=COEXPR_file_prefix,coexpr_fileSuffix=COEXPR_file_suffix,chrSizeDict=chrSizeDict, binSize = bin_size, hicFormat = "dekker")

max_coexpr_contact = max([nb_coexpr_contacts[item] for item in nb_coexpr_contacts.keys()])
max_hic_contact = max([nb_hic_contacts[item] for item in nb_hic_contacts.keys()])

# normalization
# NORMALIZATION NOT DONE HERE
print("> extract normalized Hi-C data... ")
#coexpr_contacts_norm_dict = {item:np.log2(coexpr_contacts_dict[item]*max_coexpr_contact/sum(sum(coexpr_contacts_dict[item]))+1) for item in coexpr_contacts_dict.keys()}
#hic_contacts_norm_dict = {item:np.log2(hic_contacts_dict[item]*max_hic_contact/sum(sum(hic_contacts_dict[item]))+1) for item in hic_contacts_dict.keys()}
#max_coexpr_contact_norm={item:coexpr_contacts_norm_dict[item].max() for item in coexpr_contacts_dict.keys()}
#max_hic_contact_norm={item:hic_contacts_norm_dict[item].max() for item in hic_contacts_dict.keys()}
# STILL SET THE VARIABLES BECAUSE USED IN THE FUNCTIONS
coexpr_contacts_norm_dict = coexpr_contacts_dict
hic_contacts_norm_dict = hic_contacts_dict
max_coexpr_contact={item:coexpr_contacts_dict[item].max() for item in coexpr_contacts_dict.keys()}
max_hic_contact={item:hic_contacts_dict[item].max() for item in hic_contacts_dict.keys()}


# WRITE NB CONTACT FILES
nb_coexpr_contactsFile = os.path.join(out_dir, out_dir + "_nb_coexpr_contacts.hkl")
hkl.dump(nb_coexpr_contacts, nb_coexpr_contactsFile)
print("... written: " + nb_coexpr_contactsFile)

nb_hic_contactsFile = os.path.join(out_dir, out_dir + "_nb_hic_contacts.hkl")
hkl.dump(nb_hic_contacts,nb_hic_contactsFile)
print("... written: " + nb_hic_contactsFile)


# WRITE MAX CONTACT FILES
#max_coexpr_contact_normFile = os.path.join(out_dir, out_dir + "_max_coexpr_contact_norm.hkl")
#hkl.dump(max_coexpr_contact_norm,max_coexpr_contact_normFile)
#print("... written: " + max_coexpr_contact_normFile)
#max_hic_contact_normFile = os.path.join(out_dir, out_dir + "_max_hic_contact_norm.hkl")
#hkl.dump(max_hic_contact_norm,max_hic_contact_normFile)
#print("... written: " + max_hic_contact_normFile)

max_coexpr_contact_file = os.path.join(out_dir, out_dir + "_max_coexpr_contact.hkl")
hkl.dump(max_coexpr_contact,max_coexpr_contact_file)
print("... written: " + max_coexpr_contact_file)
max_hic_contact_file = os.path.join(out_dir, out_dir + "_max_hic_contact.hkl")
hkl.dump(max_hic_contact,max_hic_contact_file)
print("... written: " + max_hic_contact_file)


# SPLIT AND WRITE TRAIN AND TEST DATA

print("> split data for training and testing... ")

#hr_mats_train,lr_mats_train,distance_train = data_split(['chr%d'%idx for idx in train_chromo_list],norm_type=2, train_data = True)
#hr_mats_test,lr_mats_test,distance_test = data_split(['chr%d'%idx for idx in test_chromo_list],norm_type=2, train_data = False)
### CHANGED HERE TO norm_type=0
hr_mats_train,lr_mats_train,distance_train = data_split(['chr%d'%idx for idx in train_chromo_list],norm_type=0, train_data = True)
hr_mats_test,lr_mats_test,distance_test = data_split(['chr%d'%idx for idx in test_chromo_list],norm_type=0, train_data = False)


train_data_file = os.path.join(out_dir, out_dir + "_train_data.hkl")
hkl.dump([lr_mats_train,hr_mats_train,distance_train], train_data_file)
print("... written: " + train_data_file)

test_data_file = os.path.join(out_dir, out_dir + "_test_data.hkl")
hkl.dump([lr_mats_test,hr_mats_test,distance_test], test_data_file)
print("... written: " + test_data_file)


#uncomment to save the raw readscount data
#hr_mats_train,lr_mats_train,distance_train = data_split(['chrom%d'%idx for idx in list(range(1,18))],norm_type=0)
#hr_mats_test,lr_mats_test,distance_test = data_split(['chrom%d'%idx for idx in list(range(18,23))],norm_type=0)
#hkl.dump([lr_mats_train,hr_mats_train,distance_train],'data/%s/train_data_raw_count.hkl'%cell)
#hkl.dump([lr_mats_test,hr_mats_test,distance_test],'data/%s/test_data_raw_count.hkl'%cell)



################################################################################################
################################################################################################ DONE
################################################################################################
endTime = datetime.datetime.now()
print("*** DONE")
print(str(startTime) + " - " + str(endTime))

