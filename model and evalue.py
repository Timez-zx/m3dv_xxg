import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K

K.set_image_dim_ordering('tf')
import keras
from mylib.models.losses import DiceLoss
from mylib.models.metrics import precision, recall, fmeasure
from keras.layers import Dense, Dropout, Flatten
from keras.layers import (Conv3D, BatchNormalization, AveragePooling3D, concatenate, Lambda, SpatialDropout3D,
                          Activation, Input, GlobalAvgPool3D, Dense, Conv3DTranspose, add)
from keras.layers import Input, Convolution3D, MaxPooling3D,AveragePooling3D, BatchNormalization,Activation
from mylib.models.metrics import invasion_acc, invasion_precision, invasion_recall, invasion_fmeasure
from keras.regularizers import l2 as l2_penalty
from keras.metrics import  binary_crossentropy
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam,SGD
from sklearn.model_selection import KFold
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

def get_gpu_session(ratio=None, interactive=False):
    config = tf.ConfigProto(allow_soft_placement=True)
    if ratio is None:
        config.gpu_options.allow_growth = True
    else:
        config.gpu_options.per_process_gpu_memory_fraction = ratio
    if interactive:
        sess = tf.InteractiveSession(config=config)
    else:
        sess = tf.Session(config=config)
    return sess


def set_gpu_usage(ratio=None):
    sess = get_gpu_session(ratio)
    K.set_session(sess)
#from mylib.models.misc import set_gpu_usage
set_gpu_usage()
#--------------------------------------------------------------------程序初始化
npz_path = '/home/ubuntu/train_and_test/sjtu-m3dv-medical-3d-voxel-classification/train_val'
npz_path1 = '/home/ubuntu/train_and_test/sjtu-m3dv-medical-3d-voxel-classification/trainx_val'
train_all_4D_voxel = np.ones((372,32,32,32,1))
test_all_4D_voxel = np.ones((93,32,32,32,1))
train_all_2D_result = np.ones((372,2))
test_all_2D_result = np.ones((93,2))
name_record0 = [0 for _ in range(465)] 
file_all = os.listdir(npz_path)
file_all.sort()
index = 0
for file in file_all:
    filelen = len(file)
    filenum = file[9:filelen-4]
    name_record0[index] = int(filenum)
    index = index+1
name_record0.sort()
index = 0
floder = KFold(n_splits=5,random_state=0,shuffle=True)
for train, test in floder.split(y_train,y_train):
    train_index = train
    test_index = test
    if index == 2:
        break
    else:
        index = index+1
#-------------------------------------------------------------------------数据分割
for index in range(0,372):
     filename = 'candidate'+str(name_record0[train_index[index]])+'.npz'
     npz_readin = np.load(os.path.join(npz_path,filename))
     temp =npz_readin['voxel']*npz_readin['seg']
     train_all_4D_voxel[index,:,:,:,0] = temp[34:66,34:66,34:66]
     train_all_2D_result[index,:] = y_train[train_index[index],:]
for index in range(0,93):
     filename = 'candidate'+str(name_record0[test_index[index]])+'.npz'
     npz_readin = np.load(os.path.join(npz_path,filename))
     temp =npz_readin['voxel']*npz_readin['seg']
     test_all_4D_voxel[index,:,:,:,0] = temp[34:66,34:66,34:66]
     test_all_2D_result[index,:] = y_train[test_index[index],:]
#——------------------------------------------------------------------------数据赋值
from sklearn.svm import LinearSVC

PARAMS = {
    'activation': lambda: Activation('relu'),  # the activation functions
    'bn_scale': True,  # whether to use the scale function in BN
    'weight_decay': 0.,  # l2 weight decay
    'kernel_initializer': 'he_uniform',  # initialization
    'first_scale': lambda x: x / 128. - 1.,  # the first pre-processing function
    'dhw': [32, 32, 32],  # the input shape
    'k': 16,  # the `growth rate` in DenseNet
    'bottleneck': 4,  # the `bottleneck` in DenseNet
    'compression': 2,  # the `compression` in DenseNet
    'first_layer': 32,  # the channel of the first layer
    'down_structure': [4, 4, 4],  # the down-sample structure
    'output_size': 2 # the output number of the classification head
}


def _conv_block(x, filters):
    bn_scale = PARAMS['bn_scale']
    activation = PARAMS['activation']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']
    bottleneck = PARAMS['bottleneck']

    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    x = Conv3D(filters * bottleneck, kernel_size=(1, 1, 1), padding='same', use_bias=False,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=l2_penalty(weight_decay))(x)
    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    x = Conv3D(filters, kernel_size=(3, 3, 3), padding='same', use_bias=True,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=l2_penalty(weight_decay))(x)
    return x


def _dense_block(x, n):
    k = PARAMS['k']

    for _ in range(n):
        conv = _conv_block(x, k)
        x = concatenate([conv, x], axis=-1)
    return x


def _transmit_block(x, is_last):
    bn_scale = PARAMS['bn_scale']
    activation = PARAMS['activation']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']
    compression = PARAMS['compression']

    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    if is_last:
        x = GlobalAvgPool3D()(x)
    else:
        *_, f = x.get_shape().as_list()
        x = Conv3D(f // compression, kernel_size=(1, 1, 1), padding='same', use_bias=True,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=l2_penalty(weight_decay))(x)
        x = AveragePooling3D((2, 2, 2), padding='valid')(x)
    return x


def get_model(weights=None, **kwargs):
    for k, v in kwargs.items():
        assert k in PARAMS
        PARAMS[k] = v
    print("Model hyper-parameters:", PARAMS)

    dhw = PARAMS['dhw']
    first_scale = PARAMS['first_scale']
    first_layer = PARAMS['first_layer']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']
    down_structure = PARAMS['down_structure']
    output_size = PARAMS['output_size']

    shape = dhw + [1]

    inputs = Input(shape=shape)

    if first_scale is not None:
        scaled = Lambda(first_scale)(inputs)
    else:
        scaled = inputs
    conv = Conv3D(first_layer, kernel_size=(3, 3, 3), padding='same', use_bias=True,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=l2_penalty(weight_decay))(scaled)

    downsample_times = len(down_structure)
    for l, n in enumerate(down_structure):
        db = _dense_block(conv, n)
        conv = _transmit_block(db, l == downsample_times - 1)

    if output_size == 1:
        last_activation = 'sigmoid'
    else:
        last_activation = 'softmax'

    outputs = Dense(output_size, activation=last_activation,
                    kernel_regularizer=l2_penalty(weight_decay),
                   kernel_initializer=kernel_initializer)(conv)
   # outputs =LinearSVC()(conv)
    model = Model(inputs, outputs)
    model.summary()

    if weights is not None:
        model.load_weights(weights, by_name=True)
    return model


def get_compiled(loss='categorical_crossentropy', optimizer='Adam',
                 metrics=["categorical_accuracy", invasion_acc,
                          invasion_precision, invasion_recall, invasion_fmeasure],
                 weights=None, **kwargs):
    model = get_model(weights=weights, **kwargs)
    model.compile(loss=loss, optimizer=optimizer,
                  metrics=[loss] + metrics)
    return model
#————————————————————————————————————————————————————————————————————————————————————训练网络
model = get_compiled()
model.fit(train_all_4D_voxel, train_all_2D_result,validation_data=(test_all_4D_voxel,test_all_2D_result),
            batch_size=16,
            epochs=30)

