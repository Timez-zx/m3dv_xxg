import keras.backend as K
import tensorflow as tf

'''
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
'''

def get_gpu_session():
    config = tf.ConfigProto(device_count = {'cuda': 0})
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    sess = tf.Session(config=config)
    return sess
    

def set_gpu_usage(ratio=None):
    sess = get_gpu_session()
    K.set_session(sess)
