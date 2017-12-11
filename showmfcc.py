

import librosa
wave, sr = librosa.load("go.wav", mono=True)
mfcc = librosa.feature.mfcc(wave, sr)
print mfcc