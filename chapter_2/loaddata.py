# -*- coding: UTF-8 -*-  
import os
import tensorflow as tf
import cv2


# 将数据打包，转换成tfrecords格式，以便后续高效读取  
def encode_to_tfrecords(tfrecord_file, data_file, is_train):

	writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_file, "dataset.tfrecords"))
	num_example = 0
	dirs = os.listdir(data_file)
	dirs.sort()
	class_number = -1
	for index, file_dir in enumerate(dirs):
		file_dirs_path = os.path.join(data_file, file_dir)
		file_dirs = os.listdir(file_dirs_path)
		file_dirs.sort()
		class_number += 1
		train_sign = -1
		for file_name in file_dirs:
			# 分离文件名与扩展名, 选择jpg格式
			if (os.path.splitext(file_name)[1] != ".jpg"): 
				continue
			train_sign += 1

			if (is_train == True and train_sign < 2): # 训练集取后面的
				continue
			if (is_train == False and train_sign >= 2): # 测试集取前面的
				break
			
			img_path = os.path.join(file_dirs_path, file_name)

			label = int(class_number)

			image = cv2.imread(img_path)
			# image = cv2.resize(image, (width, height))
			height, width, nchannel = image.shape

			example=tf.train.Example(features=tf.train.Features(feature={  
				'height':tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),  
				'width':tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),  
				'nchannel':tf.train.Feature(int64_list=tf.train.Int64List(value=[nchannel])),  
				'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),  
				'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))  
			}))  
			serialized=example.SerializeToString()
			writer.write(serialized)  

			num_example += 1  
			if (num_example % 10000 == 0):
				print(num_example)
	print(tfrecord_file, "种类数量：", class_number)
	print(tfrecord_file, "样本数据量：", num_example)
	writer.close()

# 读取tfrecords文件
def decode_from_tfrecords(filename, num_epoch=None):
	# 因为有的训练数据过于庞大，被分成了很多个文件，所以第一个参数就是文件列表名参数
	filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epoch)
	reader = tf.TFRecordReader()
	# 从 TFRecord 读取内容并保存到 serialized 中
	_,serialized = reader.read(filename_queue)
	# 读取 serialized 的格式 
	example = tf.parse_single_example(serialized, features={  
		'height':tf.FixedLenFeature([], tf.int64),  
		'width':tf.FixedLenFeature([], tf.int64),  
		'nchannel':tf.FixedLenFeature([], tf.int64),  
		'image':tf.FixedLenFeature([], tf.string),  
		'label':tf.FixedLenFeature([], tf.int64)  
	}) 
	label = tf.cast(example['label'], tf.int32)  
	image = tf.decode_raw(example['image'],tf.uint8)  
	image = tf.reshape(image, tf.stack([  
		tf.cast(example['height'], tf.int32),  
		tf.cast(example['width'], tf.int32),  
		tf.cast(example['nchannel'], tf.int32)]) )
	image = tf.cast(image, tf.float32)
	return image, label

# 根据队列流数据格式，解压出一张图片后，输入一张图片，对其做预处理、及样本随机扩充
def get_batch(image, label, batch_size, crop_size, shuffle_sign=True):
	
	distorted_image = tf.image.resize_images(image, [crop_size, crop_size])
	distorted_image = tf.random_crop(distorted_image, [crop_size, crop_size, 3])	# 随机裁剪
	# distorted_image = tf.image.random_flip_up_down(distorted_image)		# 上下随机翻转
	# distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)		# 亮度变化  
	# distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)	# 对比度变化  

	# 生成batch  
	# shuffle_batch的参数：capacity用于定义shuttle的范围，如果是对整个训练数据集，获取batch，那么capacity就应该够大  
	# 保证数据打的足够乱
	if shuffle_sign:
		images, label_batch = tf.train.shuffle_batch([distorted_image, label], batch_size = batch_size,  
												num_threads = 16, capacity = 6000, min_after_dequeue=3500) 
	else:
		images, label_batch = tf.train.batch([distorted_image, label],batch_size=batch_size)
		
	# 调试显示  
	# tf.image_summary('images', images)  
	return images, tf.reshape(label_batch, [batch_size])

# 用于测试阶段，使用的get_batch函数
def get_test_batch(image, label, batch_size, crop_size):
	distorted_image = tf.image.central_crop(image, 39./45.)  
	distorted_image = tf.random_crop(distorted_image, [crop_size, crop_size, 3])	# 随机裁剪  
	images, label_batch = tf.train.batch([distorted_image, label], batch_size = batch_size)  
	return images, tf.reshape(label_batch, [batch_size]) 

# 测试
def test():  

	encode_to_tfrecords('/home/code/lichangzhen/data/data_webface/train', '/home/code/lichangzhen/data/data_webface/webface', True)
	encode_to_tfrecords('/home/code/lichangzhen/data/data_webface/test', '/home/code/lichangzhen/data/data_webface/webface', False)
	# train_image, train_label = decode_from_tfrecords('data/train/dataset.tfrecords')
	# train_batch_image, train_batch_label = get_batch(train_image, train_label, 250, 144)		#batch 生成测试 

	# test_image, test_label   = decode_from_tfrecords('data/test/dataset.tfrecords')
	# test_batch_image, test_batch_label   = get_batch(test_image, test_label, 250, 144)		#batch 生成测试 

	# init = tf.global_variables_initializer()
	# with tf.Session() as session:  
	# 	session.run(init)  
	# 	coord = tf.train.Coordinator()  
	# 	threads = tf.train.start_queue_runners(coord=coord)  
	# 	# 每run一次，就会指向下一个样本，一直循环
	# 	for i in range(100000): 
	# 		# #image_np,label_np=session.run([image,label])#每调用run一次，那么  
	# 		# '''''cv2.imshow("temp",image_np) 
	# 		# cv2.waitKey()'''  
	# 		# #print label_np  
	# 		# #print image_np.shape  

	# 		batch_image_np,batch_label_np = session.run([batch_image,batch_label])  
	# 		# print(batch_image_np.shape)
	# 		# print(batch_label_np.shape)  
	# 		# input()
	# 		if (i % 20 == 0):
	# 			print(i)

	# 	coord.request_stop()		# queue需要关闭，否则报错  
	# 	coord.join(threads)  

# test()

