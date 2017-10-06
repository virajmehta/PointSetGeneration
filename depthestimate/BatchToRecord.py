import sys
import numpy as np
import random
import math
import os
import time
import zlib
import socket
import threading
import Queue
import sys
import cPickle as pickle
import tensorflow as tf

FETCH_BATCH_SIZE=32
BATCH_SIZE=32
HEIGHT=192
WIDTH=256
POINTCLOUDSIZE=16384
OUTPUTPOINTS=1024
REEBSIZE=1024
NUM_WORKERS=8



def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_records(dirname_in, dirname_out):
	record_fn = os.path.join(dirname_out, 'data.tfrecord')
	writer = tf.python_io.TFRecordWriter(record_fn)
	for fn in os.listdir(dirname_in):
		path = os.path.join(dirname_in, fn)
		binfile=zlib.decompress(open(path,'r').read())
		p=0
		color=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*HEIGHT*WIDTH*3],dtype='uint8').reshape((FETCH_BATCH_SIZE,HEIGHT,WIDTH,3))
		p+=FETCH_BATCH_SIZE*HEIGHT*WIDTH*3
		depth=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*HEIGHT*WIDTH*2],dtype='uint16').reshape((FETCH_BATCH_SIZE,HEIGHT,WIDTH))
		p+=FETCH_BATCH_SIZE*HEIGHT*WIDTH*2
		rotmat=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*3*3*4],dtype='float32').reshape((FETCH_BATCH_SIZE,3,3))
		p+=FETCH_BATCH_SIZE*3*3*4
		ptcloud=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*POINTCLOUDSIZE*3],dtype='uint8').reshape((FETCH_BATCH_SIZE,POINTCLOUDSIZE,3))
		ptcloud=ptcloud.astype('float32')/255
		beta=math.pi/180*20
		viewmat=np.array([[
			np.cos(beta),0,-np.sin(beta)],[
			0,1,0],[
			np.sin(beta),0,np.cos(beta)]],dtype='float32')
		rotmat=rotmat.dot(np.linalg.inv(viewmat))
		for i in xrange(FETCH_BATCH_SIZE):
			ptcloud[i]=((ptcloud[i]-[0.7,0.5,0.5])/0.4).dot(rotmat[i])+[1,0,0]
		p+=FETCH_BATCH_SIZE*POINTCLOUDSIZE*3
		reeb=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*REEBSIZE*2*4],dtype='uint16').reshape((FETCH_BATCH_SIZE,REEBSIZE,4))
		p+=FETCH_BATCH_SIZE*REEBSIZE*2*4
		keynames=binfile[p:].split('\n')
		reeb=reeb.astype('float32')/65535
		for i in xrange(FETCH_BATCH_SIZE):
			reeb[i,:,:3]=((reeb[i,:,:3]-[0.7,0.5,0.5])/0.4).dot(rotmat[i])+[1,0,0]
		data=np.zeros((FETCH_BATCH_SIZE,HEIGHT,WIDTH,4),dtype='float32')
		data[:,:,:,:3]=color*(1/255.0)
		data[:,:,:,3]=depth==0
		validating=np.array([i[0]=='f' for i in keynames],dtype='float32')
		for i in range(FETCH_BATCH_SIZE):
			img_raw = data[i,:,:,:].tostring()
			ptcloud_raw = ptcloud[i,:,:].tostring()
			example = tf.train.Example(features=tf.train.Features(feature={
				'image_raw': _bytes_feature(img_raw),
				'ptcloud_raw': _bytes_feature(ptcloud_raw)
				}))
			writer.write(example.SerializeToString())
	writer.close()


if __name__=='__main__':
	dirname_in = sys.argv[1]
	dirname_out = sys.argv[2]
	write_records(dirname_in, dirname_out)


