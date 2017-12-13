import numpy as np
import tensorflow as tf
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs_cnn9"
logdir = "{}/run-{}/".format(root_logdir, now)


with np.load('data_cache.npz') as data:
    training_Y = data['training_Y']
    testing_Y = data['testing_Y']
    training_X = data['training_X']
    testing_X = data['testing_X']

    print('tr_y: {}, tr_x: {}, te_y: {}, te_x: {}'.format(
        training_Y.shape, training_X.shape, testing_Y.shape, testing_X.shape))


num_train = training_Y.shape[0]
num_test = testing_Y.shape[0]

stride_percent = 2

input_width = training_X.shape[1]
output_width = training_Y.shape[1]

conv1_fmaps = 128
conv1_ksize = input_width // 100
conv1_stride = conv1_ksize // stride_percent
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 5
conv2_stride = conv2_ksize // stride_percent
conv2_pad = "SAME"
conv2_dropout_rate = 0.25

conv3_fmaps = 32
conv3_ksize = 5
conv3_stride = conv3_ksize // stride_percent
conv3_pad = "SAME"
conv3_dropout_rate = 0.5

pool3_size = conv2_ksize
pool3_stride = conv2_stride
pool3_fmaps = conv3_fmaps

n_fc1 = 1024
n_fc2 = 512
n_fc3 = 128
n_fc4 = 64
fc1_dropout_rate = 0.5
fc2_dropout_rate = 0.5
fc3_dropout_rate = 0.25
fc4_dropout_rate = 0.25

learning_rate = 0.01

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


reset_graph()

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, input_width], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, input_width, 1])
    y = tf.placeholder(tf.float32, shape=[None, output_width], name="y")
    training = tf.placeholder_with_default(False, shape=[], name='training')


conv1 = tf.layers.conv1d(X_reshaped, filters=conv1_fmaps,
                         kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")
conv2 = tf.layers.conv1d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")

conv3 = tf.layers.conv1d(conv2, filters=conv3_fmaps, kernel_size=conv3_ksize,
                         strides=conv3_stride, padding=conv3_pad,
                         activation=tf.nn.relu, name="conv3")

with tf.name_scope("pool3"):
    pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=pool3_size,
                                    strides=pool3_stride, padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * int(pool3.shape[1])])
    pool3_flat_drop = tf.layers.dropout(pool3_flat, conv2_dropout_rate,
                                        training=training)

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat_drop, n_fc1, activation=tf.nn.relu,
                          name="fc1")
    fc1_drop = tf.layers.dropout(fc1, fc1_dropout_rate, training=training)

with tf.name_scope("fc2"):
    fc2 = tf.layers.dense(fc1_drop, n_fc2, activation=tf.nn.relu,
                          name="fc2")
    fc2_drop = tf.layers.dropout(fc2, fc2_dropout_rate, training=training)

with tf.name_scope("fc3"):
    fc3 = tf.layers.dense(fc2_drop, n_fc3, activation=tf.nn.relu,
                          name="fc3")
    fc3_drop = tf.layers.dropout(fc3, fc3_dropout_rate, training=training)

with tf.name_scope("fc4"):
    fc4 = tf.layers.dense(fc3_drop, n_fc4, activation=tf.nn.relu,
                          name="fc4")
    fc4_drop = tf.layers.dropout(fc4, fc4_dropout_rate, training=training)

with tf.name_scope("output"):
    logits = tf.layers.dense(fc4_drop, output_width, activation=tf.sigmoid,
                             name="output")

with tf.name_scope("train"):
    loss = tf.squared_difference(logits, y)
    total_loss = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(total_loss)

with tf.name_scope("eval"):
    rounded = tf.round(logits, name='rounded')
    correct = tf.abs(rounded - y)
    accuracy = 1 - tf.reduce_mean(correct)

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


# The get_model_params() function gets the model's state (i.e., the value
# of all the variables), and the restore_model_params() restores a previous
# state. This is used to speed up early stopping: instead of storing the
# best model found so far to disk, we just save it to memory.
# At the end of training, we roll back to the best model found.

def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}


def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)


#  This implementation of Early Stopping works like this:
#   - every 100 training iterations, it evaluates the model on
#       the validation set,
#   - if the model performs better than the best model found so far,
#      then it saves the model to RAM,
#   - if there is no progress for 100 evaluations in a row,
#      then training is interrupted,
#   - after training, the code restores the best model found.

n_epochs = 50
batch_size = 30

best_loss_val = np.inf
check_interval = 500
checks_since_last_progress = 0
max_checks_without_progress = 20
best_model_params = None

mse_summary = tf.summary.scalar('MSE', total_loss)
accuracy_summary = tf.summary.scalar('Accuracy', accuracy)
file_write = tf.summary.FileWriter(logdir, tf.get_default_graph())
step = 0

with tf.Session() as sess:
    init.run()

    conv1_val = conv1.eval(feed_dict={X: [testing_X[0]], y: [testing_Y[0]]}).shape
    conv2_val = conv2.eval(feed_dict={X: [testing_X[0]], y: [testing_Y[0]]}).shape
    conv3_val = conv3.eval(feed_dict={X: [testing_X[0]], y: [testing_Y[0]]}).shape
    pool3_val = pool3_flat.eval(feed_dict={X: [testing_X[0]], y: [testing_Y[0]]}).shape

    acc_val = accuracy.eval(feed_dict={X: [testing_X[0]], y: [testing_Y[0]]})
    rounded_val = rounded.eval(feed_dict={X: [testing_X[0]], y: [testing_Y[0]]})
    correct_val = correct.eval(feed_dict={X: [testing_X[0]], y: [testing_Y[0]]})
    logits_val = logits.eval(feed_dict={X: [testing_X[0]], y: [testing_Y[0]]}).shape
    loss_val = loss.eval(feed_dict={X: [testing_X[0]], y: [testing_Y[0]]})
    total_loss_val = total_loss.eval(feed_dict={X: [testing_X[0]], y: [testing_Y[0]]})

    acc_val = accuracy.eval(feed_dict={X: testing_X, y: testing_Y})
    print("pre-training testing accuracy: {:.4f}%".format(acc_val * 100))
    for epoch in range(n_epochs):

        for iteration in range(num_train // batch_size):
            X_batch = training_X[(iteration * batch_size):((iteration + 1) * batch_size)]
            y_batch = training_Y[(iteration * batch_size):((iteration + 1) * batch_size)]
            sess.run(training_op,
                     feed_dict={X: X_batch, y: y_batch, training: True})
            if iteration % check_interval == 0:
                loss_val = total_loss.eval(feed_dict={X: testing_X,
                                                      y: testing_Y})

                mse_summary_str = mse_summary.eval(
                    feed_dict={X: testing_X, y: testing_Y})
                accuracy_summary_str = accuracy_summary.eval(
                    feed_dict={X: testing_X, y: testing_Y})
                step += 1
                file_write.add_summary(mse_summary_str, step)
                file_write.add_summary(accuracy_summary_str, step)

                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    checks_since_last_progress = 0
                    best_model_params = get_model_params()
                else:
                    checks_since_last_progress += 1
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: testing_X,
                                           y: testing_Y})
        print("Epoch {}: train accuracy: {:.4f}%\ntest accuracy: {:.4f}%\nbest loss: {:.6f}\n".format(
                  epoch, acc_train * 100, acc_val * 100, best_loss_val))
        if checks_since_last_progress > max_checks_without_progress:
            print("Early stopping!")
            break

    if best_model_params:
        restore_model_params(best_model_params)
    acc_test = accuracy.eval(feed_dict={X: testing_X,
                                        y: testing_Y})
    print("Final accuracy on test set:", acc_test)
    save_path = saver.save(sess, "./model/model")

file_write.close()
