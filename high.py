import os
import re
import sys
from glob import iglob
import numpy as np
import mfccinv
A_path =  '/media/santosh/OS_Install/Games/speech/data/Full/AB/'
#keep
filesin = os.listdir(A_path)
i = 0
new_path = '/media/santosh/OS_Install/Games/speech/data/Full/mfcctrain/'
for file in filesin:

	#os.rename(A_path+file, A_path+"A_"+str(i)+".wav")
#	print file
#	print file.split(".")[0]

	#txtfile = open(new_path+file.split(".")[0]+".txt","w")
	mfcc,inv_filter = mfccinv.createmfcc(A_path+file)
	a = np.array(mfcc)
	
	np.save(new_path+file.split('.')[0], a)
	
	
	


