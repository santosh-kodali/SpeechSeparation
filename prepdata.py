
import os
import re
import sys
from pydub import AudioSegment


A_path = '/media/santosh/Data/speech/data/A/'
B_path = '/media/santosh/Data/speech/data/B/'

full_A =  '/media/santosh/Data/speech/data/Full/A/'
full_AB =  '/media/santosh/Data/speech/data/Full/AB/'

train_A =  '/media/santosh/Data/speech/data/train/A/'
train_AB =  '/media/santosh/Data/speech/data/train/AB/'

test_A =  '/media/santosh/Data/speech/data/test/A/'
test_AB =  '/media/santosh/Data/speech/data/test/AB/'


inA = os.listdir(A_path)
inB = os.listdir(B_path)
counter = 0
for A in inA:
	for B in inB:
		if not A.endswith(".wav"): continue
		if not B.endswith(".wav"): continue

		sound1 = AudioSegment.from_file(A_path+A)
		sound2 = AudioSegment.from_file(B_path+B)

		temp1 = A.split('_')[1]
		temp1 = temp1.split('.')[0]

		combined = sound1.overlay(sound2)
		counter += 1
		name = "AB_" +temp1+"_" +str(counter)
	
		combined.export(full_AB+name, format='wav')



