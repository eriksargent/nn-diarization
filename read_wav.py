import scipy.io.wavfile as wavfile
import numpy as np

# modFunction is a list of functions that can be used to modify
# the data and rate
def read_wav(filename, time_delta=0.5, modFunctions=None):
    rate, data = wavfile.read(filename)
    if modFunctions is not None:
        # apply each mod function
        for func in modFunctions:
            # each mod function must have this signature
            newRate = func(rate, data)
            if newRate is not None:
                # modify rate if changed in mod function
                rate = newRate
    samplesPerPoint = int(rate * time_delta)
    padAmount = samplesPerPoint - (data.shape[0] % samplesPerPoint)
    # setup result data padding with zero
    res = np.zeros((data.shape[0]+padAmount,data.shape[1]), dtype=data.dtype)
    # set data in res (for padding)
    res[:data.shape[0],:data.shape[1]] = data
    # return each channel separately
    return res[:, 0].reshape(-1,samplesPerPoint), res[:, 1].reshape(-1,samplesPerPoint)


# N = np.size(data, 0)
#
# strideT = strideT / 1000    # 10 ms stride
# strideSamp = (int)(strideT * rate)
#
# # np.zeros(np.array([1,4]))
# chan1 = []
# chan2 = []
# for i in range(N):
#     if i % ((int)(strideSamp / 2)) == 0:        # assuming 50% overlap
#         # We need to see if this is the right way of getting average
#         avg1 = np.average(data[:, 0][i:(i + strideSamp)])
#         avg2 = np.average(data[:, 1][i:(i + strideSamp)])
#
#         chan1avg.append(avg1)
#         chan2avg.append(avg2)
