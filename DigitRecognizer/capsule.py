### Description ###
'''
Conv1:
    Input -- [28 x 28 x 1]
    Output -- [9 x 9 x 256]
    
    256 convolution kernals (9 x 9)
    stride = 1
    actication = ReLU
    
Primary Capsules:
    Input -- [9 x 9 x 256]
    Output -- [6 x 6 x 32]

    32 channels of convolutional capsules
    each capsule:
        8 convolutional units/kernals (9 x 9)
        stride = 2
'''

### Libraries ###
import tensorflow as tf

### Parameters ###
m_plus = 0.9 # m+
m_minus = 0.1 # m-
r = 3 # routing iterations
batch_size = 128
regularization_scale = 0.5

### Formulas ###
def softmax(logits, axis = None):
    return tf.nn.softmax(logits, axis = axis)

def squash(s_j):
    # vector s_j = sum_i(c_ij * u_j|i)
    squared = tf.reduce_sum(tf.square(s_j), -2, keepdims = True)
    result = squared / (1 + squared) * vector / tf.sqrt(squared + 1e-9)
    return result

def loss(vector, Y):
    # vector: instantiation vector of a capsule
    pass

### Layers ###
