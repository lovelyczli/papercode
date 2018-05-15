from testwebface import *


def draw_picture(data_array, path):
    scipy.misc.imsave(path, data_array)


def visualize_filters(sess, layer):
    feature = sess.run(layer)
    height, width, channel, filters = feature.shape

    if channel == 3:
        for f in range(filters):
            filte = feature[:,:,:,f]
            draw_picture(filte, 'data/filters/layer1_f'+str(f)+'.png')
    else:
        # for f in range(filters):
        #     for c in range(channel):
        #         filte = feature[:,:,c,f]
        #         draw_picture(filte, 'data/filters/layer2_f'+str(f)+'_c'+str(c)+'.png')

        f = 2
        for c in range(channel):
            filte = feature[:,:,c,f]
            draw_picture(filte, 'data/filters/layer2_f'+str(f)+'_c'+str(c)+'.png')
