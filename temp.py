import librosa
import wave
import os
full_pathA =  "/media/santosh/Data/speech/data/Full/A/A_1_.wav"

wave, sr = librosa.load(full_pathA, mono=True)
print sr
mfcc = librosa.feature.mfcc(wave, sr)
print mfcc