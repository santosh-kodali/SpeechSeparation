#Uses librosa.load parameters
import librosa
i=0
d = librosa.get_duration(filename='go.wav')
while i<d-0.1:
    wave, sr = librosa.load("go.wav", offset = i, duration = 0.1, mono=True)
    mfcc = librosa.feature.mfcc(wave, sr)
    print '\nNext duration'
    print mfcc
    i = i + 0.1