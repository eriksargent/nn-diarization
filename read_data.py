import sys
import os

import numpy as np

from read_csv import read_csv
from read_wav import read_wav

if len(sys.argv) != 2:
    raise ValueError('Please pass in the path to the data folder as a commandline argument')

data_path = sys.argv[1]

csv_path = os.path.join(data_path, 'CSV_Files_Final')
# wav_path = os.path.join(data_path, 'Sound Files')
wav_path = os.path.join(data_path, 'Downsampled')

speaker_datas = []
sound_datas = []

# Read in the data from the csv files
csv_dir = os.fsencode(csv_path)
for file in sorted(os.listdir(csv_dir)):
    filename = os.fsdecode(file)
    file_path = os.path.join(csv_path, filename)
    speaker_datas.append(read_csv(file_path))

# Flatten into an array of (time segments x samples)
speakers = np.concatenate(np.array(speaker_datas))
# And try to clean up some memory because this is a lot of data...
del speaker_datas

# Read in the data from the wav files
wav_dir = os.fsencode(wav_path)
for file in sorted(os.listdir(wav_dir)):
    filename = os.fsdecode(file)
    file_path = os.path.join(wav_path, filename)
    sound_datas.append(read_wav(file_path))
    print('Imported: {}'.format(filename))

# Flatten into an array of (time segments x samples)
# These indexes will directly match the speakers
sounds = np.concatenate(np.array(sound_datas).reshape([-1]))
# And try to clean up some memory because this is a lot of data...
del sound_datas

assert(sounds.shape[0] == speakers.shape[0])

num_samples = sounds.shape[0]

# Get some random indicies to partition the data into training and testing sets
training_size = int(0.9 * num_samples)
shuffle_indicies = np.random.permutation(num_samples)
training_idx = shuffle_indicies[:training_size]
testing_idx = shuffle_indicies[training_size:]

training_Y = speakers[training_idx, :]
testing_Y = speakers[testing_idx, :]
del speakers

training_X = sounds[training_idx, :]
testing_X = sounds[testing_idx, :]
del sounds

# Cache the data to the disk so this processing does not need to happen
# every single time we run through the network
np.savez('data_cache.npz', training_Y=training_Y, testing_Y=testing_Y,
         training_X=training_X, testing_X=testing_X)
