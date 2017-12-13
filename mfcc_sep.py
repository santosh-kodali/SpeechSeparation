#!/usr/bin/env python
#!/usr/bin/env python
#!/usr/bin/python
import numpy as np
import tensorflow as tf

import layer
import speech_data
from speech_data import Source,Target

learning_rate = 0.00001
training_iters = 300000 #steps
batch_size = 64

height=20 # mfcc features
width= 20 # (max) length of utterance
classes=4 # direction

batch = word_batch = speech_data.mfcc_batch_generator(batch_size, source=Source.DIGIT_WAVES, target=Target.direction)
X, Y = next(batch)
print("batch shape " + str(np.array(X).shape))

shape=[-1, height, width, 1]


# BASELINE toy net
def simple_dense(net): # best with lr ~0.001
	# type: (layer.net) -> None
	# net.dense(hidden=200,depth=8,dropout=False) # BETTER!!
	# net.reshape(shape)  # Reshape input picture
	net.dense(400, activation=tf.nn.tanh)# 0.99 YAY
	# net.denseNet(40, depth=4)
	# net.classifier() # auto classes from labels
	return

def alex(net): # kinda
	# type: (layer.net) -> None
	print("Building Alex-net")
	net.reshape(shape)  # Reshape input picture
	# net.batchnorm()
	net.conv([3, 3, 1, 64]) # 64 filters
	net.conv([3, 3, 64, 128])
	net.conv([3, 3, 128, 256])
	net.conv([3, 3, 256, 512])
	net.conv([3, 3, 512, 1024])
	net.dense(1024,activation=tf.nn.relu)
	net.dense(1024,activation=tf.nn.relu)


# Densely Connected Convolutional Networks https://arxiv.org/abs/1608.06993  # advanced ResNet
def denseConv(net):
	# type: (layer.net) -> None
	print("Building dense-net")
	net.reshape(shape)  # Reshape input picture
	net.buildDenseConv(nBlocks=1)
	net.classifier() # auto classes from labels


def recurrent(net):
	# type: (layer.net) -> None
	net.rnn()
	net.classifier()


def denseNet(net):
	# type: (layer.net) -> None
	print("Building fully connected pyramid")
	net.reshape(shape)  # Reshape input picture
	net.fullDenseNet()
	net.classifier() # auto classes from labels


# width=64 # for pcm baby data
# batch=speech_data.spectro_batch_generator(1000,target=speech_data.Target.digits)
# classes=10

# CHOSE MODEL ARCHITECTURE HERE:
# net=layer.net(simple_dense, data=batch,input_shape=[height,width],output_width=classes, learning_rate=learning_rate)
# net=layer.net(model=alex,input_width= width*height,output_width=classes, learning_rate=learning_rate)
# net=layer.net(model=denseConv,input_width= width*height,output_width=classes, learning_rate=learning_rate)
net = layer.net(recurrent, batch, input_shape=[height, width], output_width=classes, learning_rate=learning_rate)

# net.train(data=batch,batch_size=10,steps=500,dropout=0.6,display_step=1,test_step=1) # debug
net.train(data=batch,batch_size=10,steps=training_iters,dropout=0.6,display_step=10,test_step=100) # test
# net.train(data=batch,batch_size=batch_size,steps=training_iters,dropout=0.6,display_step=10,test_step=100) # run

# net.predict() # nil=random
# net.generate(3)  # nil=random

