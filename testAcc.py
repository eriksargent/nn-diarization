import numpy as np
import tensorflow as tf
from read_wav import read_wav
from read_csv import read_csv
import sys
import os


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


reset_graph()

model_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'model')
wav_dir_path = sys.argv[1]
csv_dir_path = sys.argv[2]

allCNNOutput = []
allExpectedOutput = []

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(os.path.join(model_path, 'model.meta'))
    saver.restore(sess, os.path.join(model_path, 'model'))

    wav_dir_enc = os.fsencode(wav_dir_path)
    for file in sorted(os.listdir(wav_dir_enc)):
        filename = os.fsdecode(file)
        if not filename.endswith('.wav'):
            continue
        print(filename)
        ch1, ch2 = read_wav(os.path.join(
            wav_dir_path, filename), time_delta=0.5)
        ch1 = ch1 / 32768
        ch2 = ch2 / 32768
        cnn_out_ch1 = sess.run(
            'eval/rounded:0', feed_dict={'inputs/X:0': ch1}).reshape([-1])
        cnn_out_ch2 = sess.run(
            'eval/rounded:0', feed_dict={'inputs/X:0': ch2}).reshape([-1])
        allCNNOutput.append(cnn_out_ch1)
        allCNNOutput.append(cnn_out_ch2)

csv_dir_enc = os.fsencode(csv_dir_path)
for file in sorted(os.listdir(csv_dir_enc)):
    filename = os.fsdecode(file)
    if not filename.endswith('.csv'):
        continue
    expData = read_csv(os.path.join(csv_dir_path, filename))
    allExpectedOutput.append(expData)

outBeforeProc = np.array(allCNNOutput).reshape([-1])
for i in range(len(allCNNOutput)):
    eliminateLessThanChunks(allCNNOutput[i], 0, 9)
    eliminateLessThanChunks(allCNNOutput[i], 1, 5)
outAfterProc = np.array(allCNNOutput).reshape([-1])
expOut = np.array(allExpectedOutput).reshape([-1])

print(outBeforeProc.shape)
print(outAfterProc.shape)
print(expOut.shape)

n = len(outBeforeProc)
assert(n == len(outAfterProc))
assert(n == len(expOut))

numCorrectBeforeProc = 0
numCorrectAfterProc = 0
for i in range(n):
    if outBeforeProc[i] == expOut[i]:
        numCorrectBeforeProc += 1
    if outAfterProc[i] == expOut[i]:
        numCorrectAfterProc += 1

print('Accuracy Before Processing: {}'.format(numCorrectBeforeProc/n*100))
print('Accuracy After Processing: {}'.format(numCorrectAfterProc/n*100))
