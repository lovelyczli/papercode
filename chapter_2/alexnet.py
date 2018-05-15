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
    W_conv1_1 = tf.get_variable('conv1_1', shape=[11, 11, 3, 96], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv1_1 = bias_variable([96])
    output1_1  = tf.nn.relu( batch_norm(conv2d(x,W_conv1_1,4,padding='VALID') + b_conv1_1))
    output1_2  = max_pool(output1_1, 3, 2, "pool1")

    # print(output1_2.shape)

    W_conv2_1 = tf.get_variable('conv2_1', shape=[5, 5, 96, 256], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv2_1 = bias_variable([256])
    output2_1  = tf.nn.relu( batch_norm(conv2d(output1_2,W_conv2_1,1) + b_conv2_1))
    output2_2  = max_pool(output2_1, 3, 2, "pool2")

    # print(output2_2.shape)

    W_conv3_1 = tf.get_variable('conv3_1', shape=[3, 3, 256, 384], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_1 = bias_variable([384])
    output3_1  = tf.nn.relu( batch_norm(conv2d(output2_2,W_conv3_1,1) + b_conv3_1))

    # print(output3_1.shape)

    W_conv4_1 = tf.get_variable('conv4_1', shape=[3, 3, 384, 384], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_1 = bias_variable([384])
    output4_1  = tf.nn.relu( batch_norm(conv2d(output3_1,W_conv4_1,1) + b_conv4_1))

    # print(output4_1.shape)

    W_conv5_1 = tf.get_variable('conv5_1', shape=[3, 3, 384, 256], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_1 = bias_variable([256])
    output5_1  = tf.nn.relu( batch_norm(conv2d(output4_1,W_conv5_1,1) + b_conv5_1))
    output5_2  = max_pool(output5_1, 3, 2, "pool5")

    # print(output5_2.shape)

    # # output = tf.contrib.layers.flatten(output)
    output5 = tf.reshape(output5_2,[-1,6*6*256])

    # print(output5.shape)

    W_fc1 = tf.get_variable('fc6', shape=[6*6*256,4096], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc1 = bias_variable([4096])
    output6_1 = tf.nn.relu( batch_norm(tf.matmul(output5,W_fc1) + b_fc1) )
    output6_2  = tf.nn.dropout(output6_1,keep_prob)

    # print(output6_2.shape)
    
    W_fc2 = tf.get_variable('fc7', shape=[4096,512], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc2 = bias_variable([512])
    output7_1 = tf.nn.relu( batch_norm(tf.matmul(output6_2,W_fc2) + b_fc2) )
    output7_2  = tf.nn.dropout(output7_1,keep_prob)

    # print(output7_2.shape)

    W_fc3 = tf.get_variable('fc8', shape=[512,10575], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc3 = bias_variable([10575])
    output = tf.matmul(output7_2,W_fc3) + b_fc3

    return output, W_conv1_1, W_conv2_1