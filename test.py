import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
import keras
from mylib.models.losses import DiceLoss
from mylib.models.metrics import precision, recall, fmeasure
from keras.layers import Dense, Dropout, Flatten
from keras.layers import (Conv3D, BatchNormalization, AveragePooling3D, concatenate, Lambda, SpatialDropout3D,
                          Activation, Input, GlobalAvgPool3D,  Conv3DTranspose, add)
from keras.layers import  Convolution3D, MaxPooling3D
from mylib.models.metrics import invasion_acc, invasion_precision, invasion_recall, invasion_fmeasure
from keras.regularizers import l2 as l2_penalty
from keras.metrics import  binary_crossentropy
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam,SGD
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from sklearn.svm import LinearSVC
from keras.models import load_model
import csv
import argparse
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

#________________________________________________________________________________
weight_path = './weight_used'
npz_path ='./test'
#______________________________________________________________________________________指定路径
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

    model = Model(inputs, outputs)
    model.summary()

    if weights is not None:
        model.load_weights(weights, by_name=True)
    return model


def get_compiled(loss='categorical_crossentropy', optimizer='adam',
                 metrics=["categorical_accuracy", invasion_acc,
                          invasion_precision, invasion_recall, invasion_fmeasure],
                 weights=None, **kwargs):
    model = get_model(weights=weights, **kwargs)
    model.compile(loss=loss, optimizer=optimizer,
                  metrics=[loss] + metrics)
    return model
#---------------------------------------------------------------模型定义
model1 = get_compiled()
model2 = get_compiled()
model3 = get_compiled()
model5 = get_compiled()
model6 = get_compiled()
model1.load_weights(os.path.join(weight_path,'weights_0693.h5'))
model2.load_weights(os.path.join(weight_path,'weights_06935.h5'))
model3.load_weights(os.path.join(weight_path,'weights_0694.h5'))
model5.load_weights(os.path.join(weight_path,'weights_07017.h5'))
model6.load_weights(os.path.join(weight_path,'weights_07035.h5'))
#——————————————————————————————————————————————————————————————————————————加载模型
file_all = os.listdir(npz_path)
file_all.sort()
test_data = np.ones((117,32,32,32,1))
name_record = [0 for _ in range(117)] 
index = 0
for file in file_all:
    filelen = len(file)
    filenum = file[9:filelen-4]
    name_record[index] = int(filenum)
    index = index+1
name_record.sort()

for index in range(0,117):
     filename = 'candidate'+str(name_record[index])+'.npz'
     npz_readin = np.load(os.path.join(npz_path,filename))
     temp =npz_readin['voxel']*npz_readin['seg']
     test_data[index,:,:,:,0] = temp[34:66,34:66,34:66]
#——————————————————————————————————————————————————————————————————————————————加载测试集
result = 0.2*(0.5*model1.predict(test_data)+0.9*model2.predict(test_data)+0.2*model3.predict(test_data))+0.4*(0.6*model5.predict(test_data)+0.4*model6.predict(test_data))
#——————————————————————————————————————————————————————————————————————————————————预测结果
predict_data = result

csv_maker = open('./final.csv','w',encoding='utf-8',newline = '')
csv_writer = csv.writer(csv_maker)
csv_writer.writerow(["Id","Predicted"])

for index in range(0,117):
    filename = 'candidate'+str(name_record[index])
    csv_writer.writerow([filename,predict_data[index,1]])
csv_maker.close()
#---------------------------------------------------------------------------------------输出结果
