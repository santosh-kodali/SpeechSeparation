import mfccinv

mfcc,inv_filter = mfccinv.createmfcc("LJ001-0001.wav")
print mfcc.shape
