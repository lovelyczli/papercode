from testwebface import *

train_flag = True

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32 )
    return tf.Variable(initial)

def conv2d(x, W, stride,padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding=padding)

def max_pool(input, k_size=1, stride=1, name=None):
    return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1], padding='VALID',name=name)

def batch_norm(input):
    return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3, is_training=train_flag, updates_collections=None)


def inference(x,train_flag_,keep_prob):
# build_network

    train_flag = train_flag_

    W_conv1_1 = tf.get_variable('conv1_1', shape=[11, 11, 3, 64], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv1_1 = bias_variable([64])
    output1  = tf.nn.relu( batch_norm(conv2d(x,W_conv1_1,2) + b_conv1_1))

    W_conv2_1 = tf.get_variable('conv2_1', shape=[3, 3, 64, 128], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv2_1 = bias_variable([128])
    output2  = tf.nn.relu( batch_norm(conv2d(output1,W_conv2_1,2) + b_conv2_1))

    W_conv3_1 = tf.get_variable('conv3_1', shape=[3, 3, 128, 256], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_1 = bias_variable([256])
    output3  = tf.nn.relu( batch_norm(conv2d(output2,W_conv3_1,2) + b_conv3_1))

    W_conv4_1 = tf.get_variable('conv4_1', shape=[3, 3, 256, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_1 = bias_variable([512])
    output4  = tf.nn.relu( batch_norm(conv2d(output3,W_conv4_1,2) + b_conv4_1))

    W_conv5_1 = tf.get_variable('conv5_1', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_1 = bias_variable([512])
    output5  = tf.nn.relu( batch_norm(conv2d(output4,W_conv5_1,2) + b_conv5_1))

    # output = tf.contrib.layers.flatten(output)
    output6 = tf.reshape(output5,[-1,8*8*512])

    W_fc1 = tf.get_variable('fc1', shape=[8*8*512,512], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc1 = bias_variable([512])
    output7 = tf.nn.relu( batch_norm(tf.matmul(output6,W_fc1) + b_fc1) )
    # output  = tf.nn.dropout(output,keep_prob)
    
    W_fc2 = tf.get_variable('fc2', shape=[512,10575], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc2 = bias_variable([10575])
    output = tf.matmul(output7,W_fc2) + b_fc2

    return output, W_conv1_1, W_conv2_1
