import numpy as np
import tensorflow as tf
from read_wav import read_wav
from read_csv import read_csv
import sys


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


reset_graph()

# with np.load('data_cache2.npz') as data:
#     training_Y = data['training_Y']
#     testing_Y = data['testing_Y']
#     training_X = data['training_X']
#     testing_X = data['testing_X']
#
#     print('tr_y: {}, tr_x: {}, te_y: {}, te_x: {}'.format(
#         training_Y.shape, training_X.shape, testing_Y.shape, testing_X.shape))

c1, c2 = read_wav(sys.argv[1], time_delta=0.5)
c1 = c1 / 32768
c2 = c2 / 32768

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./nn_erik_model/model.meta')
    saver.restore(sess, './nn_erik_model/model')
    res_ch1 = sess.run(
        'eval/rounded:0', feed_dict={'inputs/X:0': c1}).reshape([-1])
    res_ch2 = sess.run(
        'eval/rounded:0', feed_dict={'inputs/X:0': c2}).reshape([-1])

def eliminateLessThanChuncks(data, chunkVal, threshold):
    counter = 0
    for i in range(len(data)):
        if data[i] == chunkVal:
            counter += 1
        else:
            if counter < threshold:
                for j in range(counter):
                    data[i-j-1] = 1 - chunkVal
            counter = 0

eliminateLessThanChuncks(res_ch1,0,9)
eliminateLessThanChuncks(res_ch1,1,5)
eliminateLessThanChuncks(res_ch2,0,9)
eliminateLessThanChuncks(res_ch2,1,5)

res_ch1_train = read_csv(sys.argv[3]).reshape([-1])
numIncorrect = 0
numDataPoints = len(res_ch1)
assert(numDataPoints == len(res_ch1_train))
for i in range(len(res_ch1_train)):
    if res_ch1_train[i] != res_ch1[i]:
        numIncorrect += 1

print('accuracy: {}'.format((numDataPoints-numIncorrect)/numDataPoints*100))

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

xmax = len(res_ch1) * 0.05


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

ch1_intervals = createIntervalList(res_ch1)
ch2_intervals = createIntervalList(res_ch2)

textGridOutput = textGridHeader.format(xmax=xmax)
textGridOutput += itemHeader.format(itemNum=1,xmax=xmax,intervalSize=len(ch1_intervals))
for interval in ch1_intervals:
    textGridOutput += interval
textGridOutput += itemHeader.format(itemNum=2,xmax=xmax,intervalSize=len(ch2_intervals))
for interval in ch2_intervals:
    textGridOutput += interval
textGridOutput += '\n'

with open(sys.argv[2], 'w') as file:
    file.write(textGridOutput)
