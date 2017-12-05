import numpy as np
import math

def read_csv(filename, time_step=0.05, time_delta=0.5):
    input_data = np.genfromtxt(
        filename, dtype=float, delimiter=',', names=True)

    start_time = 0
    end_time = np.max(input_data['tmax'])
    duration = end_time - start_time

    samplesPerPoint = int(time_delta / time_step)
    numDataPoints = math.ceil(duration / time_delta)

    speaker_data = np.zeros((numDataPoints, samplesPerPoint), dtype=int)

    active_index = 0
    sample = 0
    reachedEnd = False
    for i in range(numDataPoints):
        for j in range(samplesPerPoint):
            if active_index >= len(input_data['text']):
                speaker_data[i,j] = 0
                continue
            if (i * samplesPerPoint + j) * time_step - start_time >= input_data['tmax'][active_index]:
                active_index += 1
            if active_index >= len(input_data['text']):
                speaker_data[i,j] = 0
                continue
            speaker_data[i,j] = input_data['text'][active_index]

    return speaker_data
