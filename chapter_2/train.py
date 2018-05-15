# -*- coding:utf-8 -*-  

import time
import math
import scipy.misc
import numpy as np
import tensorflow as tf
from loadface import *
from vggnet import *
from pyheatmap.heatmap import HeatMap

# 3641 165 7283
iterations      = 3641
batch_size      = 128
crop_size       = 256
total_epoch     = 200
weight_decay    = 0.0003
dropout_rate    = 0.5
momentum_rate   = 0.9
log_save_path   = './vgg_logs'
model_save_path = './model/test'
shuffle_sign    = True


restore = True
ckpt = 40

train_switch    = True

# ========================================================== #
# ├─ _random_crop() 
# ├─ _random_flip_leftright()
# ├─ data_augmentation()
# ├─ data_preprocessing()
# └─ learning_rate_schedule()
# ========================================================== #


def data_preprocessing(x_train,x_test):

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train[:,:,:,0] = (x_train[:,:,:,0] - np.mean(x_train[:,:,:,0])) / np.std(x_train[:,:,:,0])
    x_train[:,:,:,1] = (x_train[:,:,:,1] - np.mean(x_train[:,:,:,1])) / np.std(x_train[:,:,:,1])
    x_train[:,:,:,2] = (x_train[:,:,:,2] - np.mean(x_train[:,:,:,2])) / np.std(x_train[:,:,:,2])

    x_test[:,:,:,0] = (x_test[:,:,:,0] - np.mean(x_test[:,:,:,0])) / np.std(x_test[:,:,:,0])
    x_test[:,:,:,1] = (x_test[:,:,:,1] - np.mean(x_test[:,:,:,1])) / np.std(x_test[:,:,:,1])
    x_test[:,:,:,2] = (x_test[:,:,:,2] - np.mean(x_test[:,:,:,2])) / np.std(x_test[:,:,:,2])

    return x_train, x_test

def learning_rate_schedule(epoch_num):
    
    return 0.1 * (0.96 ** epoch_num) 


def run_testing(sess,ep):
    acc = 0.0
    loss = 0.0
    pre_index = 0
    add = 165
    for it in range(165):
        # batch_x = test_x[pre_index:pre_index+add]
        # batch_y = test_y[pre_index:pre_index+add]
        # pre_index = pre_index + add
        batch_x, batch_y = sess.run([test_batch_image, test_batch_labels])
        # batch_x, batch_y = sess.run([train_batch_image, train_batch_labels])
        loss_, acc_  = sess.run([cross_entropy,accuracy],feed_dict={x:batch_x, y_:batch_y, keep_prob: 1.0, train_flag: False})
        loss += loss_ / 165.0
        acc += acc_ / 165.0
    summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=loss), 
                            tf.Summary.Value(tag="test_accuracy", simple_value=acc)])
    return acc, loss, summary


# ========================================================== #
# ├─ main()
# Training and Testing 
# Save train/teset loss and acc for visualization
# Save Model in ./model
# ========================================================== #

def generate_heatmap(array, percentile=0.5, sign=''):
    per = np.percentile(array, 1-percentile)
    height, width = array.shape
    hdata = []
    for i in range(height):
        for j in range(width):
            if (array[i][j] > per):
                hdata.append([i,j])
    hm = HeatMap(hdata)
    hm.clickmap(save_as='hit'+str(sign)+'.png')
    hm.heatmap(save_as='heat'+str(sign)+'.png')

def generate_picture(array, sign=''):
    # scipy.misc 会自动标准化图像
    amin, amax = array.min(), array.max()
    array = (array-amin)/(amax-amin)*255
    array.astype(np.uint8)
    scipy.misc.imsave('picture'+str(sign)+'.png', array)

def generate_path_picture(array, path):
    # scipy.misc 会自动标准化图像
    amin, amax = array.min(), array.max()
    array = (array-amin)/(amax-amin)*255
    array.astype(np.uint8)
    scipy.misc.imsave(path, array)

def get_feature(sess, image, layer, filter):
    feature = sess.run(layer, feed_dict={x:np.expand_dims(image, axis = 0), keep_prob: 1.0, train_flag: False})[0]
    return feature[:,:,filter]

differ = []
def modify_midlayer(sess, image, layer, filter=0, block_size=1, stride=1, percentile=0.1):
    height, width, _ = image.shape
    feature = get_feature(sess, image, layer, filter)
    # print(feature.shape, feature)
    
    details = True

    i,ii,count = 0,0,0
    while (i + block_size - 1 < height):
        j,jj = 0,0
        while (j + block_size - 1 < width):

            temp = image.copy()
            temp[i:i+block_size,j:j+block_size,:] = 255

            tem_feature = get_feature(sess, temp, layer, filter)
            differ_graph = feature - tem_feature
            differ.append(differ_graph)

            count += 1
            if (details and count % 10 == 0):
                scipy.misc.imsave('differ/'+str(ii)+'_'+str(jj)+'.png', temp)
                generate_path_picture(differ_graph, 'differ/'+str(ii)+'_'+str(jj)+'diff'+'.png')

            print("%.2f %%" % float(i*100/height), end='\r')
            j += stride
            jj += 1
        i += stride
        ii += 1

    res = sum(differ)
    # print(type(res))
    generate_picture(res, 'sum_differ')
    # generate_heatmap(res, percentile, 1)


def modify_acc(sess, image, label, block_size=1, stride=1, percentile=0.1):
    height, width, _ = image.shape
    feature = sess.run(prob, feed_dict={x:np.expand_dims(image, axis = 0), keep_prob: 1.0, train_flag: False})[0]
    print('the example predict is ', np.equal(np.argmax(feature), label))
    # print(feature.shape, feature)
    #import pdb
    #pdb.set_trace()
    h = math.ceil((height-block_size)/stride)+1
    w = math.ceil((width-block_size)/stride)+1
    hdata = np.zeros((h, w))
    i,ii = 0,0
    while (i + block_size - 1 < height):
        j,jj = 0,0
        while (j + block_size - 1 < width):

            temp = image.copy()
            temp[i:i+block_size,j:j+block_size,:] = 255

            tem_feature = sess.run(prob, feed_dict={x:np.expand_dims(temp, axis = 0), keep_prob: 1.0, train_flag: False})[0]
            p = tem_feature[label]

            predict = np.argmax(tem_feature)
            # print(label, predict, p)
            # print(tem_feature)
            # input()
            hdata[ii][jj] = p

            print("%.2f %%" % float(i*100/height), end='\r')
            j += stride
            jj += 1
        i += stride
        ii += 1

    # import pdb
    # pdb.set_trace()
    generate_path_picture(hdata, 'acc/'+str(block_size)+'_'+str(stride)+'.png')
    generate_picture(hdata,'acc')
    # generate_heatmap(hdata, percentile)

def visual_filter(sess, layer, channel):
    feature = sess.run(layer)
    # print(len(feature), type(feature[0]), feature[0].shape)
    # input()
    height, width, chanel, filt = feature.shape
    # print(feature.shape)
    if chanel == 3:
        for i in range(filt):
            filter_ = feature[:,:,:,i]
            generate_path_picture(filter_, 'filter/'+layer.name+'__'+str(i)+'_'+'.png')
    else:
        for i in range(filt):
            for j in range(chanel):
                filter_ = feature[:,:,j,i]
                generate_path_picture(filter_, 'filter/'+layer.name+'__'+str(i)+'_'+str(j)+'.png')

if __name__ == '__main__':

    train_image, train_label = decode_from_tfrecords('/home/code/lichangzhen/data/data_shuffle/vggnet/train/dataset.tfrecords')
    train_batch_image, train_batch_label = get_batch(train_image, train_label, batch_size, crop_size, shuffle_sign)        #batch 生成测试 
    train_batch_labels = tf.one_hot(train_batch_label, 10575)
    test_image, test_label   = decode_from_tfrecords('/home/code/lichangzhen/data/data_shuffle/vggnet/test/dataset.tfrecords')
    test_batch_image, test_batch_label   = get_batch(test_image, test_label, batch_size, crop_size, shuffle_sign)       #batch 生成测试 
    test_batch_labels  = tf.one_hot(test_batch_label, 10575)


    # define placeholder x, y_ , keep_prob, learning_rate
    x  = tf.placeholder(tf.float32,[None, crop_size, crop_size, 3])
    y_ = tf.placeholder(tf.float32, [None, 10575])
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)

    output, filt1, filt2 = inference(x,train_flag,keep_prob)

    # output = alexnet_v2(inputs=x,
    #            num_outputs=10575,
    #            dropout_keep_prob=keep_prob,
    #            is_training=train_flag
    #            # reuse=False,
    #            # scope=None
    #            )

    prob = tf.nn.softmax(output)

    # loss function: cross_entropy
    # train_step: training operation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate,use_nesterov=True).minimize(cross_entropy + l2 * weight_decay)
    
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    # initial an saver to save model
    saver = tf.train.Saver(max_to_keep=20)

    # config = tf.ConfigProto(allow_soft_placement=True,
    #                         log_device_placement=False)
    # config.gpu_options.allow_growth = True
    # config=config
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())


        coord = tf.train.Coordinator()  
        threads = tf.train.start_queue_runners(coord=coord)
        if train_switch:

            summary_writer = tf.summary.FileWriter(log_save_path,sess.graph)
            # epoch = 164 
            # make sure [bath_size * iteration = data_set_number]
            ep = 0
            for _ in range(1,total_epoch+1):
                ep += 1
                if restore:
                    restore = False
                    saver.restore(sess, model_save_path + str(ckpt))
                    ep = ckpt
                    print('strat from epoch %d' % (ckpt+1) )
                    continue
                lr = learning_rate_schedule(ep)
                pre_index = 0
                train_acc = 0.0
                train_loss = 0.0
                start_time = time.time()

                print("\nepoch %d/%d:" %(ep,total_epoch))

                for it in range(1,iterations+1):
                    # batch_x = train_x[pre_index:pre_index+batch_size]
                    # batch_y = train_y[pre_index:pre_index+batch_size]

                    # batch_x = data_augmentation(batch_x)
                    batch_x, batch_y = sess.run([train_batch_image, train_batch_labels])

                    # print((sess.run(output5_2,feed_dict={x:batch_x, y_:batch_y, keep_prob: dropout_rate, learning_rate: lr, train_flag: True})).shape)
                    # input()
                    _, batch_loss = sess.run([train_step, cross_entropy],feed_dict={x:batch_x, y_:batch_y, keep_prob: dropout_rate, learning_rate: lr, train_flag: True})
                    batch_acc = accuracy.eval(feed_dict={x:batch_x, y_:batch_y, keep_prob: 1.0, train_flag: True})

                    train_loss += batch_loss
                    train_acc  += batch_acc
                    pre_index  += batch_size

                    if it == iterations:
                        train_loss /= iterations
                        train_acc /= iterations

                        loss_, acc_  = sess.run([cross_entropy,accuracy],feed_dict={x:batch_x, y_:batch_y, keep_prob: 1.0, train_flag: True})
                        train_summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=train_loss), 
                                              tf.Summary.Value(tag="train_accuracy", simple_value=train_acc)])

                        val_acc, val_loss, test_summary = run_testing(sess,ep)

                        summary_writer.add_summary(train_summary, ep)
                        summary_writer.add_summary(test_summary, ep)
                        summary_writer.flush()

                        print("iteration: %d/%d, cost_time: %ds, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f" %(it, iterations, int(time.time()-start_time), train_loss, train_acc, val_loss, val_acc))
                    else:
                        print("iteration: %d/%d, train_loss: %.4f, train_acc: %.4f" %(it, iterations, train_loss / it, train_acc / it) , end='\r')

                if ep % 5 == 0:
                    save_path = saver.save(sess, model_save_path + str(ep))
                    print("Model saved in file: %s" % save_path)  

        else:
            saver.restore(sess, model_save_path + str(ckpt))
            print('load model complete')

            shuffle_sign = False
            batch_x, batch_y = sess.run([test_batch_image, test_batch_labels])

            val_loss, val_acc = sess.run([cross_entropy,accuracy],feed_dict={x:batch_x, y_:batch_y, keep_prob: 1.0, train_flag: False})
            print('val_loss: %f, val_acc: %f' %(val_loss, val_acc))

            # for i in range(100):

            #     im = batch_x[i]
            #     label = np.argmax(batch_y[i])

            #     scipy.misc.imsave('current_picture.png', im)
            #     print('label ', label)
            #     input()

            # im = scipy.misc.imread('example.png')
            # label = 5132

            start = time.time()

            visual_filter(sess=sess, layer=filt1, channel=0)
            # modify_midlayer(sess=sess, image=im, layer=output1, filter=10, block_size=15, stride=1, percentile=0.05)
            # print('/****************************************************************************/')
            # for i in range(1,50):
            #     for j in range(1,10):
            #         modify_acc(sess=sess, image=im, label=label, block_size=i, stride=j, percentile=0.01)

            start = time.time() - start
            print("Total Time = %f" % (start))

            
        coord.request_stop()
        coord.join(threads)


    