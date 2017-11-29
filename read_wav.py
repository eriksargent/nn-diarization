import scipy.io.wavfile as wavfile
import numpy as np

def read_wav(filename, strideTms):      # strideTms is the stride in mili seconds
    rate, data = wavfile.read(filename)
    
    return data[:,0], data[:,1]             # returning channels that have been averaged  





N = np.size(data,0)

strideT = strideT / 1000    # 10 ms stride
strideSamp = (int)(strideT * rate);

# np.zeros(np.array([1,4]))
chan1 = []
chan2 = [];
for i in range(N):
  if i%((int)(strideSamp / 2)) == 0:        # assuming 50% overlap
    # We need to see if this is the right way of getting average
    avg1 = np.average(data[:,0][i:(i+strideSamp)])
    avg2 = np.average(data[:,1][i:(i+strideSamp)])

    chan1avg.append(avg1)
    chan2avg.append(avg2)

    
    




