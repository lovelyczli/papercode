from .tf_cnnvis import activation_visualization
from .tf_cnnvis import deconv_visualization
from .tf_cnnvis import deepdream_visualization

from .utils import convert_into_grid
from .utils import image_normalization

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

from six.moves import range
from six import string_types
from skimage.restoration import denoise_tv_bregman
from .utils import *
from .utils import config


import os
import time
import numpy as np
import tensorflow as tf



is_Registered = False # prevent duplicate gradient registration
# map from keyword to layer type
dict_layer = {'r' : "relu", 'p' : 'maxpool', 'c' : 'conv2d'}
units = None

configProto = tf.ConfigProto(allow_soft_placement = True)

def deepdream_visualization(sess_graph_path, value_feed_dict, layer, classes, input_tensor = None, path_logdir = './Log', path_outdir = "./Output"):
    if isinstance(layer, list):
        print("Please only give classification layer name for reconstruction.")
        return False
    elif layer in dict_layer.keys():
        print("Please only give classification layer name for reconstruction.")
        return False
    else:
        global units
        units = classes
        is_success = _get_visualization(sess_graph_path, value_feed_dict, input_tensor = input_tensor, layers = layer, method = "deepdream",
            path_logdir = path_logdir, path_outdir = path_outdir)
    return is_success

def _deepdream(graph, sess, op_tensor, X, feed_dict, layer, path_outdir, path_logdir):
    tensor_shape = op_tensor.get_shape().as_list()

    with graph.as_default() as g:
        n = (config["N"] + 1) // 2
        feature_map = tf.placeholder(dtype = tf.int32)
        tmp1 = tf.reduce_mean(tf.multiply(tf.gather(tf.transpose(op_tensor),feature_map),tf.diag(tf.ones_like(feature_map, dtype = tf.float32))), axis = 0)
        tmp2 = 1e-3 * tf.reduce_mean(tf.square(X), axis = (1, 2 ,3))
        tmp = tmp1 - tmp2
        t_grad = tf.gradients(ys = tmp, xs = X)[0]

        with sess.as_default() as sess:
            input_shape = sess.run(tf.shape(X), feed_dict = feed_dict)
            tile_size = input_shape[1 : 3]
            channels = input_shape[3]

            lap_in = tf.placeholder(np.float32, name='lap_in')
            laplacian_pyramid = lap_normalize(lap_in, channels, scale_n=config["NUM_LAPLACIAN_LEVEL"])

            image_to_resize = tf.placeholder(np.float32, name='image_to_resize')
            size_to_resize = tf.placeholder(np.int32, name='size_to_resize')
            resize_image = tf.image.resize_bilinear(image_to_resize, size_to_resize)

            end = len(units)
            for k in range(0, end, n):
                c = n
                if k + n > end:
                    c = end - ((end // n) * n)
                img = np.random.uniform(size = (c, tile_size[0], tile_size[1], channels)) + 117.0
                feed_dict[feature_map] = units[k : k + c]

                for octave in range(config["NUM_OCTAVE"]):
                    if octave > 0:
                        hw = np.float32(img.shape[1:3])*config["OCTAVE_SCALE"]
                        img = sess.run(resize_image, {image_to_resize : img, size_to_resize : np.int32(hw)})

                        for i, im in enumerate(img):
                            min_img = im.min()
                            max_img = im.max()
                            temp = denoise_tv_bregman((im - min_img) / (max_img - min_img), weight = config["TV_DENOISE_WEIGHT"])
                            img[i] = (temp * (max_img - min_img) + min_img).reshape(img[i].shape)

                    for j in range(config["NUM_ITERATION"]):
                        sz = tile_size
                        h, w = img.shape[1:3]
                        sx = np.random.randint(sz[1], size=1)
                        sy = np.random.randint(sz[0], size=1)
                        img_shift = np.roll(np.roll(img, sx, 2), sy, 1)
                        grad = np.zeros_like(img)
                        for y in range(0, max(h-sz[0]//2,sz[0]), sz[0] // 2):
                            for x in range(0, max(h-sz[1]//2,sz[1]), sz[1] // 2):
                                    feed_dict[X] = img_shift[:, y:y+sz[0],x:x+sz[1]]
                                    try:
                                        grad[:, y:y+sz[0],x:x+sz[1]] = sess.run(t_grad, feed_dict=feed_dict)
                                    except:
                                        pass

                        lap_out = sess.run(laplacian_pyramid, feed_dict={lap_in:np.roll(np.roll(grad, -sx, 2), -sy, 1)})
                        img = img + lap_out
                is_success = write_results(img, (layer, units, k), path_outdir, path_logdir, method = "deepdream")
                print("%s -> featuremap completed." % (", ".join(str(num) for num in units[k:k+c])))
    return is_success

def _visualization_by_layer_name(graph, value_feed_dict, input_tensor, layer_name, method, path_logdir, path_outdir):
    """
    Generate and store filter visualization from the layer which has the name layer_name

    :param graph:
        TF graph
    :type graph: tf.Graph object

    :param value_feed_dict:
        Values of placeholders to feed while evaluting.
        dict : {placeholder1 : value1, ...}.
    :type value_feed_dict: dict or list

    :param input_tensor:
        Where to reconstruct
    :type input_tensor: tf.tensor object (Default = None)

    :param layer_name:
        Name of the layer to visualize
    :type layer_name: String

    :param path_logdir:
        <path-to-log-dir> to make log file for TensorBoard visualization
    :type path_logdir: String (Default = "./Log")

    :param path_outdir:
        <path-to-dir> to save results into disk as images
    :type path_outdir: String (Default = "./Output")

    :return:
        True if successful. False otherwise.
    :rtype: boolean
    """
    start = -time.time()
    is_success = True

    sess = tf.get_default_session()
    if not(graph is sess.graph):
        print('Error, the graph input is not the graph of the current session!!')
    # try:
    parsed_tensors = parse_tensors_dict(graph, layer_name, value_feed_dict)
    if parsed_tensors == None:
        return is_success

    op_tensor, x, X_in, feed_dict = parsed_tensors

    is_deep_dream = True
    #is_valid_sess = True
    with graph.as_default():
        # computing reconstruction
        X = X_in
        if input_tensor != None:
            X = get_tensor(graph = graph, name = input_tensor.name)
        # original_images = sess.run(X, feed_dict = feed_dict)

        results = None
        if method == "act":
            # compute activations
            results = _activation(graph, sess, op_tensor, feed_dict)
        elif method == "deconv":
            # deconvolution
            results = _deconvolution(graph, sess, op_tensor, X, feed_dict)
        elif method == "deepdream":
            # deepdream
            is_success = _deepdream(graph, sess, op_tensor, X, feed_dict, layer_name, path_outdir, path_logdir)
            is_deep_dream = False

    # except:
    #   is_success = False
    #   print("No Layer with layer name = %s" % (layer_name))
    #   return is_success

    if is_deep_dream:
        is_success = write_results(results, layer_name, path_outdir, path_logdir, method = method)

    start += time.time()
    print("Reconstruction Completed for %s layer. Time taken = %f s" % (layer_name, start))

    return is_success

def _register_custom_gradients():
    """
    Register Custom Gradients.
    """
    global is_Registered

    if not is_Registered:
        # register LRN gradients
        @ops.RegisterGradient("Customlrn")
        def _CustomlrnGrad(op, grad):
            return grad

        # register Relu gradients
        @ops.RegisterGradient("GuidedRelu")
        def _GuidedReluGrad(op, grad):
            return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros_like(grad))

        is_Registered = True


# save given graph object as meta file
def _save_model(graph_or_sess):
    """
    Save the given TF session at PATH = "./model/tmp-model"

    :param sess:
        TF sess
    :type sess:  tf.Session object

    :return:
        Path to saved session
    :rtype: String
    """
    if isinstance(graph_or_sess, tf.Graph):
        ops = graph_or_sess.get_operations()
        for op in ops:
            if 'variable' in op.type.lower():
                raise ValueError('Please input a frozen graph (no variables). Or pass in the session object.')

        with graph_or_sess.as_default():
            sess = tf.Session(config=configProto)

            fake_var = tf.Variable([0.0], name="fake_var")
            sess.run(tf.global_variables_initializer())
    else:
        sess=graph_or_sess

    PATH = os.path.join("model", "tmp-model")
    make_dir(path = os.path.dirname(PATH))
    saver = tf.train.Saver()
    #i should deal with the case in which sess is closed.
    saver.save(sess, PATH)

    if isinstance(graph_or_sess, tf.Graph):
        sess.close()

    return PATH + ".meta"

def _write_deepdream(images, layer, path_outdir, path_logdir):
    is_success = True

    images = _im_normlize([images])
    layer, units, k = layer
    # write into disk
    path_out = os.path.join(path_outdir, "deepdream_" + layer.lower().replace("/", "_"))
    is_success = make_dir(path_out)

    for i in range(len(images)):
        for j in range(images[i].shape[0]):
            img_save = images[i][j]
            if img_save.shape[2] == 1:
                img_save = np.squeeze(img_save, axis=2)
            imsave(os.path.join(path_out, "image_%d.png" % (units[(i * images[i].shape[0]) + j + k])), img_save, format = "png")

    # write into logfile
    path_log = os.path.join(path_logdir, layer.lower().replace("/", "_"))
    is_success = make_dir(path_log)

    with tf.Graph().as_default() as g:
        image = tf.placeholder(tf.float32, shape = [None, None, None, None])

        image_summary_t = tf.summary.image(name = "One_By_One_DeepDream", tensor = image, max_outputs = config["MAX_FEATUREMAP"])

        with tf.Session() as sess:
            summary = sess.run(image_summary_t, feed_dict = {image : np.concatenate(images, axis = 0)})
        try:
            file_writer = tf.summary.FileWriter(path_log, g) # create file writer
            # compute and write the summary
            file_writer.add_summary(summary)
        except:
            is_success = False
            print("Error occured in writting results into log file.")
        finally:
            file_writer.close() # close file writer
    return is_success

# laplacian pyramid gradient normalization
def _lap_split(img):
    '''Split the image into lo and hi frequency components'''
    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img, config["K5X5"], [1, 2, 2, 1], 'SAME')
        lo2 = tf.nn.conv2d_transpose(lo, config["K5X5"] * 4, tf.shape(img), [1, 2, 2, 1])
        hi = img-lo2
    return lo, hi
def _lap_split_n(img, n):
    '''Build Laplacian pyramid with n splits'''
    levels = []
    for i in range(n):
        img, hi = _lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]
def _lap_merge(levels):
    '''Merge Laplacian pyramid'''
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, config["K5X5"]*4, tf.shape(hi), [1,2,2,1]) + hi
    return img
def _normalize_std(img):
    '''Normalize image by making its standard deviation = 1.0'''
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img), axis = (1, 2, 3), keep_dims=True))
        return img/tf.maximum(std, config["EPS"])
def lap_normalize(img, channels, scale_n):
    '''Perform the Laplacian pyramid normalization.'''
    K5X5 = k[ : , : , None , None ] / k.sum() * np.eye(channels, dtype = np.float32)
    config["K5X5"] = K5X5
    tlevels = _lap_split_n(img, scale_n)
    tlevels = list(map(_normalize_std, tlevels))
    out = _lap_merge(tlevels)
    return out

