import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # put assigned gpu number
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.19
session = tf.Session(config=config)
import numpy as np
from tensorflow.contrib.keras.api.keras.datasets.cifar10 import load_data

def next_batch(num, data, labels):
  idx = np.arange(0 , len(data))
  np.random.shuffle(idx)
  idx = idx[:num]
  data_shuffle = [data[i] for i in idx]
  labels_shuffle = [labels[i] for i in idx]

  return np.asarray(data_shuffle), np.asarray(labels_shuffle)


(x_train, y_train), (x_test, y_test) = load_data()
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10),axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10),axis=1)

#Placeholder들
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y = tf.placeholder(tf.float32, [None, 10])
dropout_prob = tf.placeholder(tf.float32)

#핵심 Neural Network
W1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], stddev=0.01))
B1 = tf.Variable(tf.constant(0.1, shape=[64]))
H1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') + B1)
W1_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=0.01))
B1_2 = tf.Variable(tf.constant(0.1, shape=[64]))
H1_2 = tf.nn.relu(tf.nn.conv2d(H1, W1_2, strides=[1, 1, 1, 1], padding='SAME') + B1_2)
P1 = tf.nn.max_pool(H1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 256], stddev=0.01))
B2 = tf.Variable(tf.constant(0.1, shape=[256]))
H2 = tf.nn.relu(tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME') + B2)
W2_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], stddev=0.01))
B2_2 = tf.Variable(tf.constant(0.1, shape=[256]))
H2_2 = tf.nn.relu(tf.nn.conv2d(H2, W2_2, strides=[1, 1, 1, 1], padding='SAME') + B2_2)
W2_3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], stddev=0.01))
B2_3 = tf.Variable(tf.constant(0.1, shape=[256]))
H2_3 = tf.nn.relu(tf.nn.conv2d(H2_2, W2_3, strides=[1, 1, 1, 1], padding='SAME') + B2_3)
P2 = tf.nn.max_pool(H2_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#W3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 512], stddev=0.01))
#B3 = tf.Variable(tf.constant(0.1, shape=[512]))
#H3 = tf.nn.relu(tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='SAME') + B3)

#W4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 1024], stddev=0.01))
#B4 = tf.Variable(tf.constant(0.1, shape=[1024]))
#H4 = tf.nn.relu(tf.nn.conv2d(H3, W4, strides=[1, 1, 1, 1], padding='SAME') + B4)

#W5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 1024, 1024], stddev=0.01))
#B5 = tf.Variable(tf.constant(0.1, shape=[1024]))
#H5 = tf.nn.relu(tf.nn.conv2d(H4, W5, strides=[1, 1, 1, 1], padding='SAME') + B5)

W_fc1 = tf.Variable(tf.truncated_normal(shape=[8 * 8 * 256, 1024], stddev=0.01))
B_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

Flat = tf.reshape(P2, [-1, 8*8*256])
H_fc1 = tf.nn.relu(tf.matmul(Flat, W_fc1) + B_fc1)

H_fc1_drop = tf.nn.dropout(H_fc1, dropout_prob)

W_fc1_2 = tf.Variable(tf.truncated_normal(shape=[1024, 512], stddev=0.01))
B_fc1_2 = tf.Variable(tf.constant(0.1, shape=[512]))

H_fc1_2 = tf.nn.relu(tf.matmul(H_fc1_drop, W_fc1_2) + B_fc1_2)

H_fc1_2_drop = tf.nn.dropout(H_fc1_2, dropout_prob)

W_fc2 = tf.Variable(tf.truncated_normal(shape=[512, 10], stddev=0.01))
B_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
logits = tf.matmul(H_fc1_2_drop,W_fc2) + B_fc2
model = tf.nn.softmax(logits)

# cost, optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(cost)

#초기화 및 세션 실행
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# batch 설정
batch_size = 500
total_batch = int(len(x_train) / batch_size)

# accuracy
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

#트레이닝 Part
print("Training Start!")
for epoch in range(21):

    for i in range(total_batch):
        batch_xs, batch_ys = next_batch(batch_size, x_train, y_train_one_hot.eval(session=sess))

        sess.run(train_step, feed_dict={X: batch_xs, Y: batch_ys, dropout_prob: 0.75})

    print('Epoch:', '%04d' % (epoch+1), 'Train Acc.=', sess.run(accuracy, feed_dict={X: batch_xs, Y: batch_ys, dropout_prob: 1.0}))
print('Training Done!')

#테스트 시작
t_accuracy = 0.0
for i in range(10):
    batch_xs, batch_ys = next_batch(1000, x_test, y_test_one_hot.eval(session=sess))
    t_accuracy+=t_accuracy + accuracy.eval(session=sess, feed_dict={X: batch_xs, Y: batch_ys, dropout_prob: 1.0})
t_accuracy = t_accuracy / 10
print('Test Acc.=', t_accuracy)
