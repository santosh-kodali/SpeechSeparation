import numpy as np
import librosa
import soundfile as sf

hop_length = 512
num_mfcc =50
n_mel_recon = 128
n_fft_recon = 2048

def invlogamplitude(S):
    return 10.0 ** (S/10.0)

def inverse_mfcc(outfile,rate, mfcc, hop_length = 512, num_mfcc = 50, n_mel_recon = 128, n_fft_recon = 2048):
    n_mfcc = mfcc.shape[0]
    dctm = librosa.filters.dct(n_mfcc, n_mel_recon)
    mel_basis = librosa.filters.mel(rate,n_fft_recon)
    bin_scaling = 1.0 / np.maximum(0.01, np.sum(np.dot(mel_basis.T, mel_basis), axis=0))
    recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T,invlogamplitude(np.dot(dctm.T, mfcc)))
    excitation = np.random.randn(hop_length*(mfcc.shape[1]-1))
    E = librosa.stft(excitation, hop_length=hop_length)
    recon = librosa.istft(E / np.abs(E) * np.sqrt(recon_stft))
    #return recon
    librosa.output.write_wav(outfile, recon, rate, 1)


data,rate = sf.read('test.flac')
mfcc = librosa.feature.mfcc(y=data, sr=rate, n_mfcc = num_mfcc, hop_length = hop_length)

inverse_mfcc('outnew.wav',rate,mfcc)
