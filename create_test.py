import wave 
import os
import sys
from random import shuffle
from shutil import copyfile
path= "data/dataset/"
testpath = "data/dataset/test/"
files = os.listdir(path)
shuffle(files)
testcount = 0
for file in files:
	if not file.endswith(".wav"): continue
	testcount = testcount + 1	
	if (testcount % 10 == 0):
		src = path+file
		dst = testpath+file
		copyfile(src, dst)
		os.remove(src)



