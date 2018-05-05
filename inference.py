import os
import cv2
import numpy as np

import tensorflow as tf

def evaluate():
         
	test_txt = './txt/test.txt'
	
	log_dir = './tmp/finetune_alexnet/n01440764/checkpoints/model_epoch10.ckpt'
	model_path = './tmp/finetune_alexnet/n01440764/checkpoints/model_epoch10.ckpt.meta'        

	imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

	#get images directory
	current_dir = os.getcwd()
	image_dir = os.path.join(current_dir, 'data','test','')

	img_files=[]
        with open(test_txt,'r') as f:
                lines = f.readlines()
                for line in lines:
                        line = line.strip('\n')
                        name=line.split('.')[-1]
                        if  name == 'JPEG':
                                img_files.append(image_dir+line)

	#load all images
	imgs = []
	f2 = open("answer.txt",'w')
	for f in img_files:
	f2.write(f.replace('.','/').split('/')[-2]+'\n')
	imgs.append(cv2.imread(f))
	f2.close()

	with tf.Session() as sess:

		saver = tf.train.import_meta_graph(model_path)# load graph
#	        model_file = tf.train.latest_checkpoint(log_dir)
		saver.restore(sess,log_dir) # load variables 		
		print("Reading checkpoints...")
#               ckpt = tf.train.get_checkpoint_state(log_dir)
#			
#               if ckpt and ckpt.model_checkpoint_path:
#           		global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#           		saver.restore(sess, ckpt.model_checkpoint_path)
#          		print('Loading success, global_step is %s' % global_step)
#       	else:
#         		print('No checkpoint file found')
#           		return

		print('\nEvaluating......')
		graph = tf.get_default_graph()# get graph
		x = graph.get_tensor_by_name('x:0')# input image
		keep_prob = graph.get_tensor_by_name('keep_prob:0')#dropout
		score = graph.get_tensor_by_name('fc8/fc8:0')#fc8 result
		softmax = tf.nn.softmax(score)

		f = open("test.txt",'w')
		for i, image in enumerate(imgs):

			# Convert image to float32 and resize to (227x227)
			img = cv2.resize(image.astype(np.float32), (227,227))

			# Subtract the ImageNet mean
			img -= imagenet_mean

			# Reshape as needed to feed into model
			img = img.reshape((1,227,227,3))

			# feed data for test
			probs = sess.run(softmax, feed_dict={x: img, keep_prob: 1})

			f.write(str(np.argmax(probs))+'\n')							
		f.close()
			
if __name__ == '__main__':  
         evaluate()
