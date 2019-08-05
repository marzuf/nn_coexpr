
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


tl.global_flag['mode']='coexprCNN'
tl.files.exists_or_mkdir(checkpoint_dir)
tl.files.exists_or_mkdir(saveCNN_dir)
tl.files.exists_or_mkdir(log_dir)

# HARD-CODED !!!

print("!!! WARNING: hard-coded settings")

patch_size = 10 #[10 x 40 kb; intially 40: 40x10kb]

batch_size = 128
lr_init = 1e-5
beta1 = 0.9
## initialize G # MZ: COMMENTED SECTION ???
#n_epoch_init = 100 # MZ: n_epoch_init -> used only in commented section
#n_epoch_init = 1
n_epoch = 5#3000
lr_decay = 0.1
decay_every = int(n_epoch / 2)
#ni = int(np.sqrt(batch_size)) # MZ:not used

# added MZ
chromo_list = list(range(1,23))
chromo_list = list(range(1,2))

print("batch_size = " + str(batch_size))
print("lr_init = " + str(lr_init))
print("beta1 = " + str(beta1))
print("n_epoch = " + str(n_epoch))
print("lr_decay = " + str(lr_decay))
print("decay_every = " + str(decay_every))
print("chromo_list = " + str(chromo_list))


#coexpr_mats_train,hic_mats_train = training_data_split(['chr%d'%idx for idx in list(range(1,18))])

#hic_mats_train,coexpr_mats_train = hkl.load('./train_data.hkl')

#coexpr_mats_train_scaled = [item*2.0/item.max()-1 for item in coexpr_mats_train]
#hic_mats_train_scaled = [item*2.0/item.max()-1 for item in hic_mats_train]


# depending of the data used, might also hold distances
# hkl file stores hkl.dump([lr_mats_train,hr_mats_train,distance_train], train_data_file) # low resol = predictor = hic; high resol = target = coexpr
tmp = hkl.load(train_data_file)
hic_mats_train = tmp[0]
coexpr_mats_train = tmp[1]



# Generator network adopts a novel dual-stream residual architecture which contains five inner residual blocks (RBs) and an outer skip connection.
# It outputs a super resolution Hi-C sample given an insufficient sequenced Hi-C sample as input. 


# Model implementation


def coexprCNN(t_image, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("coexprCNN", reuse=reuse) as vs:
        n = InputLayer(t_image, name='in')
#        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1')
#        n = Conv2d(n, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s1/c2')
#        n = Conv2d(n, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s1/c3')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s1/c2')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s1/c3')


        n = Conv2d(n, 1, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        # n.outputs
        return n


print("... build graph")    

# PLACEHOLDERS FOR THE 
t_image_hic = tf.placeholder('float32', [batch_size, patch_size, patch_size, 1], name='input_hic')
t_target_coexpr = tf.placeholder('float32', [batch_size, patch_size, patch_size, 1], name='t_target_coexpr')  # coexprdata


# CALL TO THE MODEL
out_cnn = coexprCNN(t_image_hic, is_train=True, reuse=False)

# LOSS FUNCTION
mse_loss = tl.cost.mean_squared_error(out_cnn.outputs, t_target_coexpr, is_mean=True)

cnn_vars = tl.layers.get_variables_with_name('coexprCNN', True, True)

# VARIABLES FOR THE GRAPH
with tf.variable_scope('learning_rate'):
    lr_v = tf.Variable(lr_init, trainable=False)

# OPTIMIZATION
cnn_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=cnn_vars)

#summary variables (FOR THE OUTPUT)
tf.summary.scalar("mse_loss", mse_loss)
merged_summary = tf.summary.merge_all()

# SESSION INITIALIZATION
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
tl.layers.initialize_global_variables(sess)

#record variables for tensorboard visualization
summary_writer=tf.summary.FileWriter('%s'%graph_dir,graph=tf.get_default_graph())


###========================= train CNN (coexprCNN) =========================###

print("... start training CNN")

f_out = open('%s/train.log'%log_dir,'w')
#f_out1 = open('%s/train_detail.log'%log_dir,'w')
for epoch in range(0, n_epoch + 1):
    
    print("...... epoch: " + str(epoch) + "/" + str(n_epoch))

    ## update learning rate
    if epoch != 0 and (epoch % decay_every == 0):
        #new_lr_decay = lr_decay**(epoch // decay_every)
        new_lr_decay=1
        sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
        log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
        print(log)
    elif epoch == 0:
        sess.run(tf.assign(lr_v, lr_init))
        log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
        print(log)

    epoch_time = time.time()
    total_mse_loss, n_iter = 0, 0

    for idx in range(0, len(coexpr_mats_train)-batch_size, batch_size):

        print("...... epoch " + str(epoch) + " - batch: " + str(idx+1) + "/" + str(len(coexpr_mats_train)-batch_size))

        step_time = time.time()
        b_imgs_input = hic_mats_train[idx:idx + batch_size]  # iput: low-resol -> hic-data
        b_imgs_target = coexpr_mats_train[idx:idx + batch_size] # output: high-resol -> coexpr
        ## update coexprCNN

        
        if idx == 0:
            tmp_file = os.path.join(output_dir,  "feed_data.hkl")
            hkl.dump([b_imgs_input,b_imgs_target], tmp_file)
            print("... written: " + tmp_file)




        errMSE, _ = sess.run([mse_loss, cnn_optim], {t_image_hic: b_imgs_input, t_target_coexpr: b_imgs_target})
        ## update G

        print("Epoch [%2d/%2d] %4d time: %4.4fs, mse_loss: %.8f" % (epoch, n_epoch, n_iter, time.time() - step_time, errMSE))
        total_mse_loss += errMSE

        n_iter += 1

    log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse_loss: %.8f\n" % (epoch, n_epoch, time.time() - epoch_time, total_mse_loss / n_iter)
    print(log)
    f_out.write(log)
    #record variables
    summary=sess.run(merged_summary,{t_image_hic: b_imgs_input, t_target_coexpr: b_imgs_target})
    summary_writer.add_summary(summary, epoch)

    ## save model
    if (epoch <=5) or ((epoch != 0) and (epoch % 5 == 0)):
        tl.files.save_npz(out_cnn.all_params, name=checkpoint_dir + '/g_{}_{}.npz'.format(tl.global_flag['mode'],epoch), sess=sess)
        tl.files.save_npz(out_cnn.all_params, name=checkpoint_dir + '/d_{}_{}.npz'.format(tl.global_flag['mode'],epoch), sess=sess)




################################################################################################
################################################################################################ DONE
################################################################################################
endTime = datetime.datetime.now()
print("*** DONE")
print(str(startTime) + " - " + str(endTime))





