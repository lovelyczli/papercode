# -*- coding:utf-8 -*-
from implemention import *
import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def draw_picture(data_array, path):
    scipy.misc.imsave(path, data_array)


def draw_heatmap(data_array, path):
    plt.clf()
    ax = sns.heatmap(data_array, cmap=sns.cubehelix_palette(n_colors=12, as_cmap=True, light=1))
    plt.savefig(path)


def occlusion_experiments(sess, image, prob, label, x, keep_prob, train_flag, block_size=1, stride=1, method="123"):
    debug, debug_distance = True, 10
    height, width, _ = image.shape


    prob_feature = sess.run(prob, feed_dict={x: np.expand_dims(image, axis=0), keep_prob: 1.0, train_flag: False})[0]
    predict = np.argmax(prob_feature)
    print('the example predict is ', np.equal(predict, label))

    occlusion_pictures = []
    prob_pictures = []

    # import pdb
    # pdb.set_trace()

    for row in range(0, height, stride):
        for column in range(0, width, stride):
            half = int(block_size / 2)
            top = max(0, row-half)
            down = min(height, row+half+1)
            left = max(0, column-half)
            right = min(width, column+half+1)

            # 灰色图
            temp_picture = image.copy()
            temp_picture[top:down, left:right, :] = 127
            occlusion_pictures.append(temp_picture)

            # 概率
            temp_prob_feature = sess.run(prob, feed_dict={x: np.expand_dims(temp_picture, axis=0), keep_prob: 1.0, train_flag: False})[0]
            prob_pictures.append(temp_prob_feature)

            print("%.2f %%" % float(row*100/height), end='\r')

    amount = len(occlusion_pictures)

    if debug:
        for i in range(amount):
            if (i % debug_distance == 0):
                draw_picture(occlusion_pictures[i], 'data/debug/occlusion_'+str(int(i/height))+'_'+str(int(i%height))+'.png')


    if '2' in method:
        h, w = int(height/stride), int(width/stride)
        accuracy = np.zeros((h, w))

        for i in range(amount):
            accuracy[int(i/h), int(i%h)] = prob_pictures[i][label]
        draw_picture(accuracy, 'data/occlusion/accuracy'+'.png')
        draw_heatmap(accuracy, 'data/occlusion/accuracy_heatmap'+'.png')

    if '3' in method:
        h, w = int(height/stride), int(width/stride)
        predicts = np.zeros((h, w))

        for i in range(amount):
            temp_acc = np.equal(np.argmax(prob_pictures[i]), label) * 1

            predicts[int(i/h), int(i%h)] = temp_acc
        draw_picture(predicts, 'data/occlusion/predicts'+'.png')
        draw_heatmap(predicts, 'data/occlusion/predicts_heatmap'+'.png')


