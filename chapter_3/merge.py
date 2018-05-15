#encoding=utf-8
import os
import numpy as np
import scipy.misc
from PIL import Image
import pdb
 
if __name__=='__main__':

    files = os.listdir('data/filters')

    base = [ [],[],[],[],[],[],[],[]
             ,[],[]
           ]
    # for i in range(8):
    #     base[i][0] = np.atleast_2d(Image.open('conv0/'+files[8*i]))
    white = np.zeros((5,5))
    white[:,:]=0
    scipy.misc.imsave('data/white.png', white)

    white_row = np.zeros((5))
    white_row[:]=255

    white_col = np.zeros((59))
    white_col[:]=255

    # pdb.set_trace()
    count = len(files)
    for i in range(10):
        for j in range(0,10):

            

            
            if (i == 9 and j >= 6):
                im=Image.open('data/white.png')
                _mat=np.atleast_2d(im)
                mat=_mat.copy()
                mat[:,:]=255
            else:
                im=Image.open('data/filters/'+files[10*i+j])
                mat=np.atleast_2d(im)

            # print(base[i][0].shape, mat.base)
            # pdb.set_trace()

            if (j == 0):
                base[i] = mat
            else:
                base[i] = np.column_stack((base[i],white_row))
                base[i] = np.append(base[i],mat,axis=1)
                # print(base[i].shape)
                # input()


    basenet = base[0]

    for i in range(1,10):

        basenet = np.row_stack((basenet,white_col))
        basenet = np.append(basenet,base[i],axis=0)

    print(basenet.shape)

    scipy.misc.imsave('merge.png', basenet)

    # final_img=Image.fromarray()
    # final_img.save()