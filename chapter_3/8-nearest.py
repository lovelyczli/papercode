import random
import scipy.misc
import numpy as np
import cv2


def dist(x,y):   
    return np.sqrt(np.sum((x-y)*(x-y)))

def change(data_array):
    return cv2.cvtColor(data_array,cv2.COLOR_BGR2RGB)

total = 487168
max_distence = 2147483647.0
picture_path = '/home/code/lichangzhen/experiment/visualization/lastlayer/picture/'

labels = np.load('labels.npy')
feature_data = np.load('feature.npy')

print('data load completed')
# print(feature_data.shape)
# print(labels.shape)


stack = [[max_distence, -1], [max_distence+1, -1], [max_distence+2, -1], [max_distence+3, -1], 
         [max_distence+4, -1], [max_distence+5, -1], [max_distence+6, -1], [max_distence+7, -1] ]

# choose number
number = 4554
print (number)

for index in range(total):
    if (index == number):
        continue
    distance = dist(feature_data[number], feature_data[index])
    for i in range(8):
        if (distance < stack[i][0]):
            if (i < 7):
                stack[i+1:8] = stack[i:7]
            stack[i] = [distance, index]
            break

print('chosen pictures:\n', number, labels[number])
for i in range(8):
    
    print('n'+str(i+1)+':\t', stack[i][1], labels[stack[i][1]])

exam = scipy.misc.imread(picture_path+str(number)+'.png')
exam = change(exam)
scipy.misc.imsave('result_vgg/'+ str(number)+'_init_' +str(labels[number])+'.png', exam)

for i in range(8):
    xuhao = stack[i][1]
    pic = scipy.misc.imread(picture_path+str(xuhao)+'.png')
    pic = change(pic)
    scipy.misc.imsave('result_vgg/'+   str(number)   +'_n' + str(i+1)+'_'+str(xuhao)+'_'+str(labels[xuhao])+'.png', pic)

        


