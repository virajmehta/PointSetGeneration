import sys
import tensorflow as tf
import numpy as np
import tflearn
import random
import math
import os
os.system("chmod +w /unsullied/sharefs/wangmengdi/wangmengdi")
import time
import zlib
import socket
import threading
import Queue
import sys
import tf_nndistance
import cPickle as pickle

from BatchFetcher import *

HEIGHT=192
WIDTH=256
POINTCLOUDSIZE=16384
batch_size=32

lastbatch=None
lastconsumed=FETCH_BATCH_SIZE

def fetch_batch():
	global lastbatch,lastconsumed
	if lastbatch is None or lastconsumed+BATCH_SIZE>FETCH_BATCH_SIZE:
		lastbatch=fetchworker.fetch()
		lastconsumed=0
	ret=[i[lastconsumed:lastconsumed+BATCH_SIZE] for i in lastbatch]
	lastconsumed+=BATCH_SIZE
	return ret
def stop_fetcher():
	fetchworker.shutdown()

def model_fn(features, labels, mode, params):
    tflearn.init_graph(seed=1029,num_cores=2,gpu_memory_fraction=0.9,soft_placement=True)
    #img_inp=tf.placeholder(tf.float32,shape=(BATCH_SIZE,HEIGHT,WIDTH,4),name='img_inp')
    BATCH_SIZE = params['batch_size']
    pt_gt=tf.placeholder(tf.float32,shape=(BATCH_SIZE,POINTCLOUDSIZE,3),name='pt_gt')

    x=features['img']
#192 256
    x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x0=x
    x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#96 128
    x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x1=x
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#48 64
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x2=x
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#24 32
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x3=x
    x=tflearn.layers.conv.conv_2d(x,256,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#12 16
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x4=x
    x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#6 8
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x5=x
    x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#3 4
    x_additional=tflearn.layers.core.fully_connected(x,2048,activation='relu',weight_decay=1e-3,regularizer='L2')
    x=tflearn.layers.conv.conv_2d_transpose(x,256,[5,5],[6,8],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#6 8
    x5=tflearn.layers.conv.conv_2d(x5,256,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
    x=tf.nn.relu(tf.add(x,x5))
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x5=x
    x=tflearn.layers.conv.conv_2d_transpose(x,128,[5,5],[12,16],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#12 16
    x4=tflearn.layers.conv.conv_2d(x4,128,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
    x=tf.nn.relu(tf.add(x,x4))
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x4=x
    x=tflearn.layers.conv.conv_2d_transpose(x,64,[5,5],[24,32],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#24 32
    x3=tflearn.layers.conv.conv_2d(x3,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
    x=tf.nn.relu(tf.add(x,x3))
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x3=x
    x=tflearn.layers.conv.conv_2d_transpose(x,32,[5,5],[48,64],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#48 64
    x2=tflearn.layers.conv.conv_2d(x2,32,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
    x=tf.nn.relu(tf.add(x,x2))
    x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x2=x
    x=tflearn.layers.conv.conv_2d_transpose(x,16,[5,5],[96,128],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#96 128
    x1=tflearn.layers.conv.conv_2d(x1,16,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
    x=tf.nn.relu(tf.add(x,x1))
    x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#48 64
    x2=tflearn.layers.conv.conv_2d(x2,32,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
    x=tf.nn.relu(tf.add(x,x2))
    x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x2=x
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#24 32
    x3=tflearn.layers.conv.conv_2d(x3,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
    x=tf.nn.relu(tf.add(x,x3))
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x3=x
    x=tflearn.layers.conv.conv_2d(x,128,(5,5),strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#12 16
    x4=tflearn.layers.conv.conv_2d(x4,128,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
    x=tf.nn.relu(tf.add(x,x4))
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x4=x
    x=tflearn.layers.conv.conv_2d(x,256,(5,5),strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#6 8
    x5=tflearn.layers.conv.conv_2d(x5,256,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
    x=tf.nn.relu(tf.add(x,x5))
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x5=x
    x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#3 4
    x_additional=tflearn.layers.core.fully_connected(x_additional,2048,activation='linear',weight_decay=1e-4,regularizer='L2')
    x_additional=tf.nn.relu(tf.add(x_additional,tflearn.layers.core.fully_connected(x,2048,activation='linear',weight_decay=1e-3,regularizer='L2')))
    x=tflearn.layers.conv.conv_2d_transpose(x,256,[5,5],[6,8],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#6 8
    x5=tflearn.layers.conv.conv_2d(x5,256,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
    x=tf.nn.relu(tf.add(x,x5))
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x5=x
    x=tflearn.layers.conv.conv_2d_transpose(x,128,[5,5],[12,16],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#12 16
    x4=tflearn.layers.conv.conv_2d(x4,128,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
    x=tf.nn.relu(tf.add(x,x4))
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x4=x
    x=tflearn.layers.conv.conv_2d_transpose(x,64,[5,5],[24,32],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#24 32
    x3=tflearn.layers.conv.conv_2d(x3,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
    x=tf.nn.relu(tf.add(x,x3))
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')

    x_additional=tflearn.layers.core.fully_connected(x_additional,1024,activation='relu',weight_decay=1e-3,regularizer='L2')
    x_additional=tflearn.layers.core.fully_connected(x_additional,256*3,activation='linear',weight_decay=1e-3,regularizer='L2')
    x_additional=tf.reshape(x_additional,(BATCH_SIZE,256,3))
    x=tflearn.layers.conv.conv_2d(x,3,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
    x=tf.reshape(x,(BATCH_SIZE,32*24,3))
    x=tf.concat([x_additional,x],1)
    x=tf.reshape(x,(BATCH_SIZE,OUTPUTPOINTS,3))
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"points":x}
        )
    dists_forward,_,dists_backward,_=tf_nndistance.nn_distance(pt_gt,x)
    mindist=dists_forward
    dist0=mindist[0,:]
    dists_forward=tf.reduce_mean(dists_forward)
    dists_backward=tf.reduce_mean(dists_backward)
    loss_nodecay=(dists_forward+dists_backward/2.0)*10000
    loss=loss_nodecay+tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))*0.1
    optimizer = tf.train.AdamOptimizer(3e-5*BATCH_SIZE/FETCH_BATCH_SIZE).minimize(loss,global_step=batchno)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=loss_nodecay)

def _parse_function(example_proto):
        features = {'image_raw': tf.FixedLenFeature([], tf.string),
                    'ptcloud_raw': tf.FixedLenFeature([], tf.string)}
        parsed_features = tf.parse_single_example(example_proto, features)
        image = tf.decode_raw(parsed_features['image_raw'], tf.float32)
        image_shape = (HEIGHT, WIDTH, 4)
        image =tf.reshape(image, image_shape)
        ptcloud = tf.decode_raw(parsed_features['ptcloud_raw'], tf.float32)
        ptcloud_shape = (POINTCLOUDSIZE, 3)
        ptcloud = tf.reshape(ptcloud, ptcloud_shape)
        return {'images': image}, ptcloud

def input_fn():
    filenames = ['../dataset/data.tfrecord']
    dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    features, labels= iterator.get_next()
    return features, labels


def main():
    if not os.path.exists(dumpdir):
	   os.system("mkdir -p %s"%dumpdir)
    model_params = {'batch_size': batch_size}
    psgn = tf.estimator.Estimator(model_fn=model_fn, params=model_params)
    psgn.train(input_fn=input_fn, steps=300000)

def dumppredictions(resourceid,keyname,valnum):
	img_inp,x,pt_gt,loss,optimizer,batchno,batchnoinc,mindist,loss_nodecay,dists_forward,dists_backward,dist0=build_graph(resourceid)
	config=tf.ConfigProto()
	config.gpu_options.allow_growth=True
	config.allow_soft_placement=True
	saver=tf.train.Saver()
	fout = open("%s/%s.v.pkl"%(dumpdir,keyname),'wb')
	with tf.Session(config=config) as sess:
		#sess.run(tf.initialize_all_variables())
		sess.run(tf.global_variables_initializer())
		saver.restore(sess,"%s/%s.ckpt"%(dumpdir,keyname))
		fetchworker.bno=0
		fetchworker.start()
		cnt=0
		for i in xrange(0,300000):
			t0=time.time()
			data,ptcloud,validating=fetch_batch()
			validating=validating[0]!=0
			if not validating:
				continue
			cnt+=1
			pred,distmap=sess.run([x,mindist],feed_dict={img_inp:data,pt_gt:ptcloud})
			pickle.dump((i,data,ptcloud,pred,distmap),fout,protocol=-1)
			print i,'time',time.time()-t0,cnt
			if cnt>=valnum:
				break
	fout.close()

if __name__=='__main__':
    main()
'''resourceid = 0
	datadir,dumpdir,cmd,valnum="data","dump","predict",3
	for pt in sys.argv[1:]:
		if pt[:5]=="data=":
			datadir = pt[5:]
		elif pt[:5]=="dump=":
			dumpdir = pt[5:]
		elif pt[:4]=="num=":
			valnum = int(pt[4:])
		else:
			cmd = pt
	if datadir[-1]=='/':
		datadir = datadir[:-1]
	if dumpdir[-1]=='/':
		dumpdir = dumpdir[:-1]
	assert os.path.exists(datadir),"data dir not exists"
	os.system("mkdir -p %s"%dumpdir)
	fetchworker=BatchFetcher(datadir)
	print "datadir=%s dumpdir=%s num=%d cmd=%s started"%(datadir,dumpdir,valnum,cmd)
	
	keyname=os.path.basename(__file__).rstrip('.py')
	try:
		if cmd=="train":
			main(resourceid,keyname)
		elif cmd=="predict":
			dumppredictions(resourceid,keyname,valnum)
		else:
			assert False,"format wrong"
	finally:
		stop_fetcher()
    '''