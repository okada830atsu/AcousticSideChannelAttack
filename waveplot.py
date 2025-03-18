
import torchaudio
import matplotlib.pyplot as plt

#path = 'dataset/train/1_14.wav'
path = 'dataset/train/5_689.wav'
waveform, sample_rate = torchaudio.load(uri=path)
print(waveform.size())
plt.plot(waveform[0])
plt.show()