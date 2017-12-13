

import librosa
import numpy 



wave, sr = librosa.load("go_1.wav", mono=True)
mfcc = librosa.feature.mfcc(wave, sr)
print (mfcc.shape)

