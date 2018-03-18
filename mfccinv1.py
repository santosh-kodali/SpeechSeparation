import numpy as np
import librosa
import soundfile as sf
data,rate = sf.read('test.flac')
outfile = 'outnew.wav'
hop_length = 512
num_mfcc =20
n_mel_recon = 128
n_fft_recon = 2048

def invlogamplitude(S):
    return 10.0 ** (S/10.0)

mfcc = librosa.feature.mfcc(y=data, sr=rate, n_mfcc = 20, hop_length = hop_length)

n_mfcc = mfcc.shape[0]
dctm = librosa.filters.dct(n_mfcc, n_mel_recon)
mel_basis = librosa.filters.mel(rate,n_fft_recon)
bin_scaling = 1.0 / np.maximum(0.01, np.sum(np.dot(mel_basis.T, mel_basis), axis=0))
recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T,invlogamplitude(np.dot(dctm.T, mfcc)))
excitation = np.random.randn(hop_length*(mfcc.shape[1]-1))
E = librosa.stft(excitation, hop_length=hop_length)
recon = librosa.istft(E / np.abs(E) * np.sqrt(recon_stft))
librosa.output.write_wav(outfile, recon, rate, 1)