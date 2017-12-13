import numpy as np
import tensorflow as tf
from read_wav import read_wav
from read_csv import read_csv
import sys
import os

textGridHeader = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = {xmax:.2f}
tiers? <exists>
size = 2
item []:'''

itemHeader = '''
    item [{itemNum}]:
        class = "IntervalTier"
        name = "Channel{itemNum}"
        xmin = 0
        xmax = {xmax:.2f}
        intervals: size = {intervalSize}'''

intervalTemplate = '''
        intervals [{intervalNum}]:
            xmin = {lower:.2f}
            xmax = {upper:.2f}
            text = "{res}"'''


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def eliminateLessThanChunks(data, chunkVal, threshold):
    counter = 0
    for i in range(len(data)):
        if data[i] == chunkVal:
            counter += 1
        else:
            if counter < threshold:
                for j in range(counter):
                    data[i - j - 1] = 1 - chunkVal
            counter = 0


def createIntervalList(results):
    intervalList = []
    curRes = results[0]
    curStart = 0
    intervalNum = 1
    for i in range(len(results)):
        if results[i] != curRes:
            intervalList.append(intervalTemplate.format(intervalNum=intervalNum,
                                                        lower=curStart * 0.05, upper=i * 0.05, res='N' if curRes == 0 else 'S'))
            curStart = i
            intervalNum += 1
            curRes = results[i]
    return intervalList


reset_graph()

model_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'nn_erik_model')

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(os.path.join(model_path, 'model.meta'))
    saver.restore(sess, os.path.join(model_path, 'model'))

    wav_dir_path = sys.argv[1]
    wav_dir_enc = os.fsencode(wav_dir_path)
    for file in os.listdir(wav_dir_enc):
        filename = os.fsdecode(file)
        if not filename.endswith('.wav'):
            continue
        print(filename)
        ch1, ch2 = read_wav(os.path.join(wav_dir_path, filename), time_delta=0.5)
        ch1 = ch1 / 32768
        ch2 = ch2 / 32768
        cnn_out_ch1 = sess.run(
            'eval/rounded:0', feed_dict={'inputs/X:0': ch1}).reshape([-1])
        cnn_out_ch2 = sess.run(
            'eval/rounded:0', feed_dict={'inputs/X:0': ch2}).reshape([-1])

        # post processing
        eliminateLessThanChunks(cnn_out_ch1, 0, 9)
        eliminateLessThanChunks(cnn_out_ch1, 1, 5)
        eliminateLessThanChunks(cnn_out_ch2, 0, 9)
        eliminateLessThanChunks(cnn_out_ch2, 1, 5)

        xmax = len(cnn_out_ch1) * 0.05

        ch1_intervals = createIntervalList(cnn_out_ch1)
        ch2_intervals = createIntervalList(cnn_out_ch2)

        textGridOutput = textGridHeader.format(xmax=xmax)
        textGridOutput += itemHeader.format(itemNum=1,
                                            xmax=xmax,
                                            intervalSize=len(ch1_intervals))
        for interval in ch1_intervals:
            textGridOutput += interval
        textGridOutput += itemHeader.format(itemNum=2,
                                            xmax=xmax,
                                            intervalSize=len(ch2_intervals))
        for interval in ch2_intervals:
            textGridOutput += interval
        textGridOutput += '\n'
        with open(os.path.join(sys.argv[2], filename.replace('.wav', '.TextGrid')), 'w') as f:
            f.write(textGridOutput)
