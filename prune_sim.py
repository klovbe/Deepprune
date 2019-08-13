import glob
import h5py
import subprocess
import random
import pdb
import matplotlib.pyplot as plt
import os
import glob
from keras.layers import Layer, InputSpec
import tensorflow as tf
import keras
from keras import backend as K
import keras.callbacks
from keras import layers
# Helper libraries
import numpy as np
from data_load import *
# from prun import *
import pickle

from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
from keras.layers import Conv1D, MaxPooling1D, Dense, Activation, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.metrics import roc_auc_score, auc, roc_curve
from keras import regularizers
import sys

# read the hyper-parameters
data_path = sys.argv[1]
result_path = sys.argv[2]
data_info = sys.argv[3]
kernel_number = int(sys.argv[4])
random_seed = int(sys.argv[5])
GPU_SET = sys.argv[6]
ker_len = int(sys.argv[7])
lr_init = float(sys.argv[8])
lr_decay = int(sys.argv[9])
decay_rate = float(sys.argv[10])
spar_penal = int(sys.argv[11])
bn = int(sys.argv[12])
ridge_penal = int(sys.argv[13])
modep = int(sys.argv[14])

# data_info =3
# data_path = "/data/public/simulation/HDF5/Simulation"+str(data_info)+"/"
# result_path = "/data/wlchi/python_project/luoxiao/ePooling/ePooling-temp-data/now"
# kernel_number=256
# random_seed=0
# GPU_SET='3'
# ker_len=24
# lr_init = 0.01
# lr_decay = True
# decay_rate = 1.2
# spar_penal = False
# bn = False
# ridge_penal = False
# modep = 0




test_dataset = data_path + "test.hdf5"
training_dataset = data_path + "train.hdf5"
epoch_num = 1000
valadation_split = 0.1
patience = 15
batch_size = 256
pooling = 'max'
prun_time = 7
# set the result path
output_path = result_path + "/" + str(data_info)

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_SET
gpu_options = tf.GPUOptions(allow_growth=True)


np.random.seed(random_seed)
random.seed(random_seed)
from keras import backend as K
tf.set_random_seed(random_seed)
sess = tf.Session(graph=tf.get_default_graph(), config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=1,
                                                                      inter_op_parallelism_threads=1))
K.set_session(sess)





def mkdir(path):
    isexists = os.path.exists(path)
    if not isexists:
        os.makedirs(path)
        return True
    else:
        return False


mkdir(output_path)

X_test, Y_test, X_train, Y_train = get_data(data_path)
X_train_pos = X_train[Y_train.astype('int') == 1]
X_train_neg = X_train[Y_train.astype('int') == 0]
# if X_train.shape[0] > 5000:
#     X_compute = X_train[np.random.choice(X_train.shape[0], 5000)]
# else:
#     X_compute = X_train

print(X_test.shape)
print(Y_test.shape)
input_shape = (X_train.shape[1], 4)


def build_CNN(model_template, kernel_number, ker_len, input_shape, bn,
              pooling="GlobalMax"):
    def relu_advanced(x):
        return K.relu(x, alpha=0.5, max_value=10)

    model_template.add(Conv1D(
        input_shape=input_shape,
        kernel_size=ker_len,
        filters=kernel_number,
        padding='same',
        strides=1, name='conv'))
    if bn:
        model_template.add(layers.BatchNormalization())
    model_template.add(Activation(relu_advanced))
    model_template.add(keras.layers.GlobalMaxPooling1D(name='score'))
    if spar_penal:
        if ridge_penal:
            model_template.add(keras.layers.core.Dense(output_dim=1, kernel_regularizer=regularizers.l1_l2(),
                                                       name='Dense_l1'))
        else:
            model_template.add(
                keras.layers.core.Dense(output_dim=1, kernel_regularizer=regularizers.l1(), name='Dense_l1'))
    else:
        if ridge_penal:
            model_template.add(keras.layers.core.Dense(output_dim=1, kernel_regularizer=regularizers.l2(),
                                                       name='Dense_l1'))
        else:
            model_template.add(keras.layers.core.Dense(output_dim=1, name='Dense_l1'))
    model_template.add(keras.layers.Activation("sigmoid"))
    return model_template


def fit(output_path, kernel_number, mode, model, optimizer):
    output_prefix = output_path + "/" \
                    + "_random-seed_" + str(random_seed) \
                    + "_batch-size_" + str(batch_size) \
                    + '_kernel-length_' + str(ker_len) \
                    + '_pooling_' + str(pooling) \
                    + "model-KernelNum_" + str(kernel_number) \
                    + '_mode_' + str(mode) \
                    + '_lr-init_' + str(lr_init) \
                    + '_lr-decay_' + str(lr_decay) \
                    + '_decay-rate_' + str(decay_rate) \
                    + '_spar-penal_' + str(spar_penal) \
                    + '_bn_' + str(bn) \
                    + '_ridge-penal_' + str(ridge_penal) \
                    + '_modep_' + str(modep)

    modelsave_output_filename = output_prefix + "_checkpointer.hdf5"
    history_output_path = output_prefix + '.history'
    prediction_save_path = output_prefix + '.npy'
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    # set the checkpoint and earlystop to save the best model
    earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=patience,
                                                 verbose=1)

    checkpointer = keras.callbacks.ModelCheckpoint(filepath=modelsave_output_filename,
                                                   verbose=1,
                                                   save_best_only=True)

    # train the model, and save the history
    history = model.fit(X_train,
                        Y_train,
                        epochs=epoch_num,
                        batch_size=batch_size,
                        validation_split=valadation_split,
                        verbose=0,
                        callbacks=[checkpointer, earlystopper])

    # load the best weight
    model.load_weights(modelsave_output_filename)

    # get the prediction of the test data set, and save as .npy
    prediction = model.predict(X_test)
    np.save(prediction_save_path, prediction)

    # save the history
    with open(history_output_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    return [modelsave_output_filename, history_output_path, prediction_save_path]


def predict(output_path, kernel_number, mode, model):
    output_prefix = output_path + "/" \
                    + "_random-seed_" + str(random_seed) \
                    + "_batch-size_" + str(batch_size) \
                    + '_kernel-length_' + str(ker_len) \
                    + '_pooling_' + str(pooling) \
                    + "model-KernelNum_" + str(kernel_number) \
                    + '_mode_' + str(mode) \
                    + '_lr-init_' + str(lr_init) \
                    + '_lr-decay_' + str(lr_decay) \
                    + '_decay-rate_' + str(decay_rate) \
                    + '_spar-penal_' + str(spar_penal) \
                    + '_bn_' + str(bn) \
                    + '_ridge-penal_' + str(ridge_penal) \
                    + '_modep_' + str(modep)

    modelsave_output_filename = output_prefix + "_checkpointer.hdf5"
    prediction_save_path = output_prefix + '.npy'
    # load the best weight
    model.save_weights(modelsave_output_filename)

    # get the prediction of the test data set, and save as .npy
    prediction = model.predict(X_test)
    np.save(prediction_save_path, prediction)

    return None


for i in range(prun_time):
    print("prune time {}".format(i))
    if i == 0:
        model = keras.models.Sequential()
        model = build_CNN(model,
                          kernel_number,
                          ker_len,
                          input_shape=input_shape, bn=bn,
                          pooling=pooling)
        dense_weights_origin, dense_bias_origin = model.get_layer('Dense_l1').get_weights()
        [conv_weights_origin, conv_bias_origin] = model.get_layer('conv').get_weights()
        fit(output_path, kernel_number, mode='init', model=model, optimizer=Adam(lr=lr_init))
    else:
        # get_layer_output = K.function([model.input],
        #                                   [model.get_layer('score').output])
        # layer_output = get_layer_output([X_compute])[0]
        dense_weights, dense_bias = model.get_layer('Dense_l1').get_weights()
        [conv_weights, conv_bias] = model.get_layer('conv').get_weights()
        if modep == 0:
            dense_abs_weights = abs(dense_weights.squeeze())
            part = np.percentile(dense_abs_weights, 50)
            mask_o = dense_abs_weights > part
        elif modep == 1:
            pre_model = keras.models.Model(model.input, model.get_layer('score').output)
            layer_output_pos = pre_model.predict(X_train_pos)
            layer_output_neg = pre_model.predict(X_train_neg)
            multi_out_pos = np.mean(layer_output_pos, axis=0)
            multi_out_neg = np.mean(layer_output_neg, axis=0)
            multi_out = abs(multi_out_pos - multi_out_neg)
            part_o = np.percentile(multi_out, 50)
            mask_o = multi_out > part_o
        elif modep == 2:
            pre_model = keras.models.Model(model.input, model.get_layer('score').output)
            layer_output_pos = pre_model.predict(X_train_pos)
            layer_output_neg = pre_model.predict(X_train_neg)
            multi_out_pos = np.mean(np.multiply(layer_output_pos, dense_weights.squeeze()), axis=0)
            multi_out_neg = np.mean(np.multiply(layer_output_neg, dense_weights.squeeze()), axis=0)
            multi_out = abs(multi_out_pos - multi_out_neg)
            part_o = np.percentile(multi_out, 50)
            mask_o = multi_out > part_o
        # dense_abs_weights = abs(dense_weights.squeeze())
        # part_w = np.percentile(dense_abs_weights, 50)
        # mask_w = dense_abs_weights > part_w
        # mask = (mask_w + mask_o) > 0
        kernel_number_new = sum(mask_o)
        model = keras.models.Sequential()
        model = build_CNN(model,
                          kernel_number_new,
                          ker_len,
                          input_shape=input_shape, bn=bn,
                          pooling=pooling)
        lr = lr_init / decay_rate ** i if lr_decay else lr_init
        init_weights = model.get_weights()
        print("init decay train of {} kernel with learning rate {}".format(kernel_number_new, lr))
        fit(output_path, kernel_number_new, mode='init-decay', model=model, optimizer=Adam(lr=lr))
        if lr_decay:
            model.set_weights(init_weights)
            print("init train of {} kernel with learning rate {}".format(int(kernel_number / (2 ** i)), lr_init))
            fit(output_path, int(kernel_number / (2 ** i)), mode='init', model=model, optimizer=Adam(lr=lr_init))
        ###########################################prune reinit############################################
        dense_weights_origin = dense_weights_origin[mask_o]
        conv_weights_origin = conv_weights_origin[:, :, mask_o]
        conv_bias_origin = conv_bias_origin[mask_o]
        model.get_layer('Dense_l1').set_weights([dense_weights_origin, dense_bias_origin])
        model.get_layer('conv').set_weights([conv_weights_origin, conv_bias_origin])
        print("reinit prune train of {} kernel with learning rate {}".format(kernel_number_new, lr_init))
        fit(output_path, kernel_number_new, mode='prune-reinit', model=model, optimizer=Adam(lr=lr_init))
        ###########################################prune##########################################
        new_dense_weights = dense_weights[mask_o]
        new_conv_weights = conv_weights[:, :, mask_o]
        new_conv_bias = conv_bias[mask_o]
        model.get_layer('Dense_l1').set_weights([new_dense_weights, dense_bias])
        model.get_layer('conv').set_weights([new_conv_weights, new_conv_bias])
        predict(output_path, kernel_number_new, mode='inter', model=model)
        print("prune train of {} kernel with learning rate {}".format(kernel_number_new, lr))
        fit(output_path, kernel_number_new, mode='prune', model=model, optimizer=Adam(lr=lr))





