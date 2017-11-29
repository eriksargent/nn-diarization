import sys
import os

import numpy as np

from read_csv import read_csv
from read_wav import read_wav

if len(sys.argv) != 2:
    raise ValueError('Please pass in the path to the data folder as a commandline argument')

data_path = sys.argv[1]

csv_path = os.path.join(data_path, 'CSV_Files_Final')
wav_path = os.path.join(data_path, 'Sound Files')

speaker_datas = []
sound_datas = []

# Read in the data from the csv files
csv_dir = os.fsencode(csv_path)
for file in sorted(os.listdir(csv_dir)):
    filename = os.fsdecode(file)
    file_path = os.path.join(csv_path, filename)
    speaker_datas.append(read_csv(file_path))

# Read in the data from the wav files
wav_dir = os.fsencode(wav_path)
for file in sorted(os.listdir(wav_dir)):
    filename = os.fsdecode(file)
    file_path = os.path.join(wav_path, filename)
    sound_datas.append(read_wav(file_path))

