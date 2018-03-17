
import os
import re
import sys
from pydub import AudioSegment
from random import shuffle
import mfccinv
import soundfile as sf
import numpy as np

#/media/santosh/OS_Install/Games/speech/data/LJSpeech-1.1/wav
A_path = '/media/santosh/OS_Install/Games/speech/data/A/wavs/'
B_path = '/media/santosh/OS_Install/Games/speech/data/B/speakerb/'

full_A =  '/media/santosh/OS_Install/Games/speech/data/Full/A/'
full_AB =  '/media/santosh/OS_Install/Games/speech/data/Full/AB/'

train =  '/media/santosh/Data/speech/data/train/'
train_AB =  '/media/santosh/Data/speech/data/'

test_A =  '/media/santosh/Data/speech/data/test/A/'
test_AB =  '/media/santosh/Data/speech/data/test/AB/'


inA = os.listdir(A_path)
inB = os.listdir(B_path)
shuffle(inB)
counter = 0
counter2 =0
for A in inA:
	counter2 = 0
	if counter == 2:
		break
	for B in inB:
		counter+=1

		if not A.endswith(".wav"): continue
		if not B.endswith(".flac"): continue
		
			
		sound1 = AudioSegment.from_file(A_path+A)
		sound2 = AudioSegment.from_file(B_path+B)

		temp1 = A.split('_')[1]
		temp1 = temp1.split('.')[0]

		combined = sound1.overlay(sound2)
	
		name = "AB_" +temp1+"_" +str(counter)
		sound1.export(train_AB+"A.wav",format='wav')
		sound2.export(train_AB+"B.wav",format='wav')

		combined.export(train_AB+"testing.wav", format='wav')
		
		mfcc,inv_filter = mfccinv.createmfcc(A_path+A)
		a = np.array(mfcc)
	
		np.save(train+'A'+str(counter), a)

		mfcc,inv_filter = mfccinv.createmfcc(train+"export.wav")
		a = np.array(mfcc)
	
		np.save(train+'B_'+str(counter), a)
		
		counter2+=1
		if counter2 ==2:
			break
			



