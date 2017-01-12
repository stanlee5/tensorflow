import tensorflow as tf
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

# IMDB Dataset loading
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
                                valid_portion=0.1)

print len(train), len(train[0])

BATCH_SIZE = 1024
MAX_LENGTH = 200
VOCAB_SIZE = 10000
EMBEDDING_SIZE = 128
DATA_SIZE = len(train[0])

trainX, trainY = train
testX, testY = test
trainX = pad_sequences(trainX, MAX_LENGTH, value=.0)
trainY = to_categorical(trainY, nb_classes=2)
testX = pad_sequences(testX, MAX_LENGTH, value=.0)
testY = to_categorical(testY, nb_classes=2)

print('*** Dataset ready..!! ***')


text = tf.placeholder(tf.int32, [None, MAX_LENGTH])
label = tf.placeholder(tf.float32, [None, 2])

with tf.device("/cpu:0"):
    embedding = tf.get_variable("embedding", [VOCAB_SIZE, EMBEDDING_SIZE], dtype=tf.float32)
    inputs = tf.nn.embedding_lookup(embedding, text)

HIDDEN_NODE_1 = 128
LSTM1_f = tf.nn.rnn_cell.LSTMCell(HIDDEN_NODE_1)
LSTM1_b = tf.nn.rnn_cell.LSTMCell(HIDDEN_NODE_1)

out, state = tf.nn.dynamic_rnn(LSTM1_f, inputs, dtype=tf.float32)
#out, state = tf.nn.bidirectional_dynamic_rnn(LSTM1_f, LSTM1_b, inputs=inputs, dtype=tf.float32)

out = tf.transpose(out, [1,0,2])
last = tf.gather(out, int(out.get_shape()[0])-1)
print('val:',tf.shape(last))
print('last:',tf.shape(last))

weight = tf.Variable(tf.truncated_normal([HIDDEN_NODE_1, int(label.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[label.get_shape()[1]]))
prob = tf.nn.softmax(tf.matmul(last, weight) + bias)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(prob, label, name='Loss')
minimize = tf.train.AdamOptimizer().minimize(cross_entropy)

mistake = tf.equal(tf.argmax(label, 1), tf.argmax(prob, 1))
acc = tf.reduce_mean(tf.cast(mistake, tf.float32))


with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    Epoch = 10
    num_on_batch = int(len(trainX)/BATCH_SIZE)
    for i in xrange(Epoch):
        print 'Epoch:',i
        print sess.run(acc, {text: testX, label: testY})

        ptr = 0
        for j in xrange(num_on_batch):
            inp, outp = trainX[ptr:ptr+BATCH_SIZE], trainY[ptr:ptr+BATCH_SIZE]
            sess.run(minimize, {text: inp, label: outp})
            ptr += BATCH_SIZE

            print j, sess.run(acc, {text: testX, label: testY})
