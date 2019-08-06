
# coding: utf-8

import os, time, pickle, random, time, sys, math
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import matplotlib.pyplot as plt
import hickle as hkl


from sklearn import datasets, linear_model  
from sklearn.metrics import mean_squared_error 

import datetime

# INPUT_SPLIT_HiC_COEPXR/INPUT_SPLIT_HiC_COEPXR_nb_coexpr_contacts.hkl

#python train_hicGAN.py <TRAIN_DATA_file> <MAIN_OUTPUT_DIR> 
# python train_nnCoexpr.py INPUT_SPLIT_HiC_COEPXR/INPUT_SPLIT_HiC_COEXPR_train_data.hkl TRAIN_coexprCNN

#GPU setting and Global parameters
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6" #MZ: not used ???

#checkpoint_dir = "checkpoint"
#log_dir = "log"
#graph_dir = "graph"
#saveCNN_dir = "samples"
#checkpoint_dir = sys.argv[1]
#log_dir = sys.argv[2]
#graph_dir = sys.argv[3]
#saveCNN_dir = sys.argv[4]


startTime = datetime.datetime.now()


train_data_file = sys.argv[1]

if not os.path.exists(train_data_file):
    print("ERROR: input file does not exist !")
    sys.exit(1)

output_dir = sys.argv[2]
os.makedirs(output_dir, exist_ok = True)
checkpoint_dir = os.path.join(output_dir, "checkpoint")
log_dir = os.path.join(output_dir, "log")
graph_dir = os.path.join(output_dir, "graph")
saveCNN_dir = os.path.join(output_dir, "samples")
log_file = os.path.join(output_dir, "params_logFile.txt")
print("... write logs in:\t" + log_file)
if os.path.exists(log_file):
    os.remove(log_file)

tl.global_flag['mode']='coexprCNN'
tl.files.exists_or_mkdir(checkpoint_dir)
tl.files.exists_or_mkdir(saveCNN_dir)
tl.files.exists_or_mkdir(log_dir)

# HARD-CODED !!!

mylog = open(log_file,"a+") 
plog = "startTime\t=\t" + str(startTime)
print(plog)
mylog.write(plog + "\n")
mylog.close() 

print("!!! WARNING: hard-coded settings")

patch_size = 10 #[10 x 40 kb; intially 40: 40x10kb]

batch_size = 128
lr_init = 1e-5
beta1 = 0.9
## initialize G # MZ: COMMENTED SECTION ???
#n_epoch_init = 100 # MZ: n_epoch_init -> used only in commented section
#n_epoch_init = 1
n_epoch = 1000#3000
lr_decay = 0.1
decay_every = int(n_epoch / 2)
#ni = int(np.sqrt(batch_size)) # MZ:not used


mylog = open(log_file,"a+") 
plog = "> HARD-CODED SETTINGS:"
mylog.write(plog + "\n")
plog = "batch_size\t=\t" + str(batch_size) 
print(plog)
mylog.write(plog + "\n")
mylog.close() 


#coexpr_mats_train,hic_mats_train = training_data_split(['chr%d'%idx for idx in list(range(1,18))])

#hic_mats_train,coexpr_mats_train = hkl.load('./train_data.hkl')

#coexpr_mats_train_scaled = [item*2.0/item.max()-1 for item in coexpr_mats_train]
#hic_mats_train_scaled = [item*2.0/item.max()-1 for item in hic_mats_train]


# depending of the data used, might also hold distances
# hkl file stores hkl.dump([lr_mats_train,hr_mats_train,distance_train], train_data_file) # low resol = predictor = hic; high resol = target = coexpr
tmp = hkl.load(train_data_file)
hic_mats_train = tmp[0]
coexpr_mats_train = tmp[1]

mylog = open(log_file,"a+") 
plog = "hic_mats_train.shape\t=\t" + str(hic_mats_train.shape)
print(plog)
mylog.write(plog + "\n")
plog = "coexpr_mats_train.shape\t=\t" + str(coexpr_mats_train.shape)
print(plog)
mylog.write(plog + "\n")
mylog.close() 




tmp = hkl.load(test_data_file)
hic_mats_test = tmp[0]
coexpr_mats_test = tmp[1]

mylog = open(log_file,"a+") 
plog = "hic_mats_test.shape\t=\t" + str(hic_mats_test.shape)
print(plog)
mylog.write(plog + "\n")
plog = "coexpr_mats_test.shape\t=\t" + str(coexpr_mats_test.shape)
print(plog)
mylog.write(plog + "\n")
mylog.close() 






                                                        #### https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
                                                        ## Use only one feature
                                                        #diabetes_X = diabetes.data[:, np.newaxis, 2]

                                                        ## Split the data into training/testing sets
                                                        #diabetes_X_train = diabetes_X[:-20]
                                                        #diabetes_X_test = diabetes_X[-20:]

                                                        ## Split the targets into training/testing sets
                                                        #diabetes_y_train = diabetes.target[:-20]
                                                        #diabetes_y_test = diabetes.target[-20:]

                                                        ## Create linear regression object
                                                        #regr = linear_model.LinearRegression()

                                                        ## Train the model using the training sets
                                                        #regr.fit(diabetes_X_train, diabetes_y_train)

                                                        ## Make predictions using the testing set
                                                        #diabetes_y_pred = regr.predict(diabetes_X_test)

                                                        ## The coefficients
                                                        #print('Coefficients: \n', regr.coef_)
                                                        ## The mean squared error
                                                        #print("Mean squared error: %.2f"
                                                        #      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
                                                        ## Explained variance score: 1 is perfect prediction
                                                        #print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

                                                        ## Plot outputs
                                                        #plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
                                                        #plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

                                                        #plt.xticks(())
                                                        #plt.yticks(())

                                                        #plt.show()



###========================= fit the linear regressions =========================###

print("... start linear regression"\n)


  

all_errMSE = []

for idx in range(0, len(coexpr_mats_train)-batch_size, batch_size):

    print("..." + " - batch: " + str(idx+1) + "/" + str(len(coexpr_mats_train)-batch_size))


    b_imgs_input_X_train = hic_mats_train[idx:idx + batch_size]  # iput: low-resol -> hic-data
    b_imgs_target_Y_train = coexpr_mats_train[idx:idx + batch_size] # output: high-resol -> coexpr

    # create linear regression object
    lreg = linear_model.LinearRegression() # sklearn.linear_model

    # train the model using training set
    regr.fit(b_imgs_input_X_train, b_imgs_target_Y_train)


    ## ITERATE OVER THE TEST DATA

    assert len(hic_mats_test) == len(coexpr_mats_test)

    repMSE = 0
    i_avg = 0
    for i_rep in range(0, len(hic_mats_test)):
        i_avg += 1
        print("..." + " - batch " + str(idx+1) + "/" + str(len(coexpr_mats_train)-batch_size) + " : i_rep " + str(i_rep) + "/" + str(len(hic_mats_test) + 1) )


        b_imgs_input_X_test = hic_mats_test[i_rep][idx:idx + batch_size]  # iput: low-resol -> hic-data
        b_imgs_target_Y_test = coexpr_mats_test[i_rep][idx:idx + batch_size] # output: high-resol -> coexpr


        # make predictions for the test data # => iterate over the test data

        b_imgs_target_Y_test_PRED = regr.predict(b_imgs_input_X_test)

        errMSE = mean_squared_error(b_imgs_target_Y_test, b_imgs_target_Y_test_PRED) # sklearn.metrics.mean_squared_error

        repMSE += errMSE
        
    avgMSE = repMSE*1.0/i_avg

    all_errMSE.append(avgMSE) 

    








all_errMSE_file = os.path.join(out_dir, "all_errMSE.hkl")
hkl.dump(all_errMSE, all_errMSE_file)
print("... written: " + all_errMSE_file)






################################################################################################
################################################################################################ DONE
################################################################################################
endTime = datetime.datetime.now()
print("*** DONE")
print(str(startTime) + " - " + str(endTime))





