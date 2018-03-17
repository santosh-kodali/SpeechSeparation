import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected,reshape
from tflearn.layers.estimator import regression
import speech_data
import numpy as np
from speech_data import Source,Target
import mfccinv
train_path = "/media/santosh/Data/speech/data/"

batch_size = 200
height=80 # mfcc features
width= 250 # (max) length of utterance
print "begin 1"
test_x, test_y = speech_data.mfcc_batch_test_generator2(50)
print "between"
batch = speech_data.mfcc_batch_generator2(batch_size)
print "begin 2"
#test data here
X, Y = next(batch)


# print("batch shape " + str(np.array(X).shape))

# print("batch shape " + str(np.array(test_x).shape))

X = np.array(X).reshape([-1, height, width, 1])
#Y = np.array(Y).reshape([-1,height*width])
#test_y = np.array(test_y).reshape([-1,height*width])
test_x = np.array(test_x).reshape([-1, height, width, 1])

print("x " + str(np.array(X).shape))
print("y "+str(np.array(Y).shape))

print("testx " + str(np.array(test_x).shape))
print("testy " + str(np.array(test_y).shape))

print "begin 3"


shape=[-1, height, width, 1]

convnet = input_data(shape=[None, height, width, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 256, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

#convnet = reshape(convnet, [20,80], name='reshape')
#convnet = max_pool_2d(convnet, 5)
	
convnet = fully_connected(convnet, 2500, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 5000, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 20000)
convnet = regression(convnet, optimizer='adam', learning_rate=0.0001, loss='mean_square', name='targets')

model = tflearn.DNN(convnet)

for x in range(0,1000):
	
	model.fit({'input': X}, {'targets': Y}, n_epoch=6, validation_set=({'input': test_x}, {'targets': test_y}), 
	   snapshot_step=500, show_metric=True)
	X,Y = next(batch)
	X = np.array(X).reshape([-1, height, width, 1])
	#	model.save("model.tfl")
	
model.load("model.tfl")
a,b = mfccinv.createmfcc(train_path+"testing.wav")
a = mfccinv.inversemfcc(a,b)
"""
a = np.array(a)
a=np.pad(a,((0,0),(0,250-len(a[0]))), mode='constant', constant_values=0)

#print(a.shape())
a = np.array(a).reshape([-1,height, width, 1])

a = model.predict(a)

#a = a[0][:13040]
a = np.array(a).reshape([80,250])
np.savetxt("temp1.txt",a)
a = mfccinv.inversemfcc(a,b)


justa ,x= mfccinv.createmfcc(train_path+"A.wav")
justa = np.array(justa)
np.savetxt("temp2.txt",justa)
"""