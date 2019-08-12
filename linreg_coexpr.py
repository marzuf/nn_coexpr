
# coding: utf-8

import os, time, pickle, random, time, sys, math

import numpy as np
from time import localtime, strftime
import logging, scipy
import matplotlib.pyplot as plt
import hickle as hkl


from sklearn import datasets, linear_model  
from sklearn.metrics import mean_squared_error 

import datetime

# INPUT_SPLIT_HiC_COEPXR/INPUT_SPLIT_HiC_COEPXR_nb_coexpr_contacts.hkl

# python linreg_coexpr.py INPUT_SPLIT_HiC_COEXPR/INPUT_SPLIT_HiC_COEXPR_train_data_sub100.hkl INPUT_SPLIT_HiC_COEXPR/INPUT_SPLIT_HiC_COEXPR_test_data_sub100.hkl OUTPUT_LINREG_SUB100
# python linreg_coexpr.py <train_data_file> <test_data_file> <OUTPUT_DIR>


# train_data_file = "INPUT_SPLIT_HiC_COEXPR/INPUT_SPLIT_HiC_COEXPR_train_data_sub100.hkl"

startTime = datetime.datetime.now()

train_data_file = sys.argv[1]
test_data_file = sys.argv[2]

if not os.path.exists(train_data_file):
    print("ERROR: input train data file does not exist !")
    sys.exit(1)

if not os.path.exists(test_data_file):
    print("ERROR: input test data file does not exist !")
    sys.exit(1)

output_dir = sys.argv[3]
os.makedirs(output_dir, exist_ok = True)

log_file = os.path.join(output_dir, "params_logFile.txt")
print("... write logs in:\t" + log_file)
if os.path.exists(log_file):
    os.remove(log_file)


# HARD-CODED !!!
mylog = open(log_file,"a+") 
plog = "startTime\t=\t" + str(startTime)
print(plog)
mylog.write(plog + "\n")
mylog.close() 

print("!!! WARNING: hard-coded settings")


batch_size = 128
batch_size = 1


mylog = open(log_file,"a+") 
plog = "> HARD-CODED SETTINGS:"
mylog.write(plog + "\n")
plog = "batch_size\t=\t" + str(batch_size) 
print(plog)
mylog.write(plog + "\n")
mylog.close() 


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

print("... start linear regression\n")

all_errMSE = []

for idx in range(0, len(coexpr_mats_train)-batch_size, batch_size):

    print("..." + " - batch: " + str(idx+1) + "/" + str(len(coexpr_mats_train)-batch_size))


    b_imgs_input_X_train = hic_mats_train[idx:idx + batch_size]  # iput: low-resol -> hic-data
    b_imgs_target_Y_train = coexpr_mats_train[idx:idx + batch_size] # output: high-resol -> coexpr

    # create linear regression object
    lreg = linear_model.LinearRegression() # sklearn.linear_model

    # train the model using training set # => should be of shape (n, 1)
    lreg.fit(b_imgs_input_X_train.flatten()[:,np.newaxis], b_imgs_target_Y_train.flatten()[:,np.newaxis])


    ## ITERATE OVER THE TEST DATA

    assert len(hic_mats_test) == len(coexpr_mats_test)
    
    # ??? TEST ALL TEST DATA IN ONE RUN ???    
    
    b_imgs_input_X_test = hic_mats_test[idx:idx + batch_size]  # iput: low-resol -> hic-data
    b_imgs_target_Y_test = coexpr_mats_test[idx:idx + batch_size] # output: high-resol -> coexpr

    b_imgs_target_Y_test_PRED = lreg.predict(b_imgs_input_X_test.flatten()[:,np.newaxis])

    errMSE = mean_squared_error(b_imgs_target_Y_test.flatten()[:,np.newaxis], b_imgs_target_Y_test_PRED) # sklearn.metrics.mean_squared_error
   

    if idx % 10 == 0:
        print("... batch " + str(idx+1) + "/" + str(len(coexpr_mats_train)-batch_size) + "\t:\tMSE = " + str(errMSE))
        

    all_errMSE.append(errMSE) 

    # HARD-CODED !!!
    mylog = open(log_file,"a+") 
    plog = "idx\t=\t" + str(idx) + "; MSE\t=\t" + str(errMSE)
    print(plog)
    mylog.write(plog + "\n")
    mylog.close() 


all_errMSE_file = os.path.join(output_dir, "all_errMSE.hkl")
hkl.dump(all_errMSE, all_errMSE_file)
print("... written: " + all_errMSE_file)






################################################################################################
################################################################################################ DONE
################################################################################################
endTime = datetime.datetime.now()
print("*** DONE")
print(str(startTime) + " - " + str(endTime))

mylog = open(log_file,"a+") 
mylog.write(str(startTime) + "\n" + str(endTime))
mylog.close() 

print("... logs written in: " + log_file)

  



