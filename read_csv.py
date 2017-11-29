import numpy as np


def read_csv(filename):
    input_data = np.genfromtxt(filename, dtype=float, delimiter=',', names=True)

    # Some of the files have different names for this field...
    # start_time = np.min(input_data['tmi0'])
    # So just assume all files start at time 0
    start_time = 0
    end_time = np.max(input_data['tmax'])
    duration = end_time - start_time


    # Convert to a value every <time_step> seconds
    # This parameter will change the number of elements in the output array
    # and it should likely be tweaked to try different values
    time_step = 0.05
    samples = int(duration / time_step)

    speaker_data = np.zeros(samples)

    active_index = 0
    sample = 0
    while sample * time_step - start_time < end_time - time_step:
        if sample * time_step - start_time >= input_data['tmax'][active_index]:
            active_index += 1

        speaker_data[sample] = input_data['text'][active_index]
        sample += 1

    return speaker_data
