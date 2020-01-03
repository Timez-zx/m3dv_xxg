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

final_score = 0
final_epoch = 0
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.x_val,self.y_val = validation_data
    def on_epoch_end(self, epoch, log={}):
        global final_score
        global final_epoch
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.x_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            if score > final_score:
                final_score = score
                final_epoch = epoch+1
            print('\n ROC_AUC - score:%.6f \n' % score)
            print('\n final_score:%.6f \n' % final_score)
            print('\n final_epoch:%d \n' % final_epoch)

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
def mix_up(data,label,num_of_data,beta):
    data_index = np.round(np.random.rand(num_of_data,2)*464)
    data_index=data_index.astype(np.int)
    data_back = np.zeros((num_of_data,np.size(data,1),np.size(data,2),np.size(data,3),1))
    label_back = np.zeros((num_of_data,np.size(label,1)))
    for data_count in range(0,num_of_data):
        data_back[data_count] = beta*data[data_index[data_count,0]]+(1-beta)*data[data_index[data_count,1]]
        label_back[data_count] = beta*label[data_index[data_count,0]]+(1-beta)*label[data_index[data_count,1]]
        if data_count% 40 == 0:
            print(data_count)
    return data_back,label_back

def data_shift(data,label,mode,rotate_beta):
    data_back = np.zeros((np.size(data,0),np.size(data,1),np.size(data,2),np.size(data,3),1))
    label_back = label
    for data_count in range(0,np.size(data,0)):
        if mode == 0:   #x轴翻转
             data_back[data_count,:,:,:,0] = np.flip(data[data_count,:,:,:,0],2)
        elif mode == 1:   #y轴翻转
            data_back[data_count,:,:,:,0] = np.flip(data[data_count,:,:,:,0],1)
        elif mode == 2:   #90度旋转
             data_back[data_count,:,:,:,0] = np.rot90(data[data_count,:,:,:,0],rotate_beta)#旋转90*ratate_beta度        
    return data_back,label_back

def data_shift2(data,mode,rotate_beta):
    data_back = np.zeros((np.size(data,0),np.size(data,1),np.size(data,2)))
    for data_count in range(0,np.size(data,0)):
        if mode == 0:   #x轴翻转
             data_back = np.flip(data,2)
        elif mode == 1:   #y轴翻转
            data_back = np.flip(data,1)
        elif mode == 2:   #90度旋转
             data_back = np.rot90(data,rotate_beta)#旋转90*ratate_beta度 
        elif mode == 3:#z轴
             data_back = np.flip(data,0)
    return data_back

def set_gpu_usage(ratio=None):
    sess = get_gpu_session(ratio)
    K.set_session(sess)
#from mylib.models.misc import set_gpu_usage
set_gpu_usage()
#___________________________________________________________________________________________________训练过程中定义的所有函数
import pandas as pd                         #导入pandas包
train_all_answers0 = pd.read_csv("/home/ubuntu/train_and_test/sjtu-m3dv-medical-3d-voxel-classification/train_val.csv",usecols = [1]) 
train_all_answers = train_all_answers0.values.reshape(465,1)
BATCH_SIZE = 1
NUM_CLASSES = 2
NUM_EPOCHS = 12
y_train = keras.utils.to_categorical(train_all_answers, NUM_CLASSES)
#-----------------------------------------------------------------------------------------训练数据的读入
npz_path = '/home/ubuntu/train_and_test/sjtu-m3dv-medical-3d-voxel-classification/train_val'
train_all_4D_voxel = np.ones((930,32,32,32,1))
train_all_2D_result = np.ones((930,2))
test_data = np.ones((117,32,32,32,1))
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
#——----------------------------------------------------------------------------------------空间分配
for index in range(0,465):
     filename = 'candidate'+str(name_record0[index])+'.npz'
     npz_readin = np.load(os.path.join(npz_path,filename))
     temp =npz_readin['voxel']*npz_readin['seg']
     temp0 = data_shift2(temp,0,0)
     temp1 = data_shift2(temp0,1,0)
     temp2 = data_shift2(temp1,3,0)
     temp3 = data_shift2(temp1,2,1)
     train_all_4D_voxel[index,:,:,:,0] = temp[34:66,34:66,34:66]
     train_all_2D_result[index,:] = y_train[index,:]
     train_all_4D_voxel[465+index,:,:,:,0] = temp1[34:66,34:66,34:66]
     train_all_2D_result[465+index,:] = y_train[index,:]
#----------------------------------------------------------------------------------------------训练数据赋值
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
#-------------------------------------------------------------------------------------------训练网络使用
model = get_compiled()
model.fit(train_all_4D_voxel,train_all_2D_result,
         batch_size=16,
         epochs=30)
    