
import numpy as np
import matplotlib.pyplot as plt

a_time = 30.0
z_time = 31.0
sample_rate = 48000
data = np.fromfile("D:/tmp/chobits.pcm", dtype = '<i2')
print(data.shape)
data = data[int(a_time * sample_rate):int(z_time * sample_rate)]
# data = data[::10]
time = np.linspace(0, data.shape[0], data.shape[0])
data.tofile("D:/tmp/chobits_.pcm")
# time = np.linspace(int(0 * sample_rate), int((z_time - a_time) * sample_rate), data.shape[0])
print(f"{data.shape} {data.max()} {data.min()}")
plt.plot(time, data)
plt.title("waveform")
plt.xlabel("time")
plt.ylabel("amplitude")
plt.show()
