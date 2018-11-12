from keras import backend as K
from keras import regularizers
from keras.layers import Input, Dense, Lambda, Activation, Dropout, Flatten, Bidirectional, Conv2D, MaxPool2D, Reshape, BatchNormalization, Layer, Embedding, dot
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model, Progbar, normalize
from keras.layers.recurrent import LSTM
from keras.layers.merge import add, concatenate
import utils
import keras
import numpy as np
from datetime import datetime
import os
import sys
import json
import pickle
import gensim
from sklearn.metrics import f1_score, accuracy_score

# check args
if len(sys.argv) != 5:
    print("Usage:\npython tmn_run.py <input_data_dir> <embedding_file> <output_dir> <topic_num>")
    exit(0)
######################## configurations ########################
data_dir = sys.argv[1] #"../data/tmn"    # data dir
embedding_fn = sys.argv[2] #"/data1/jichuanzeng/workspace/glove.6B.200d.txt"
output_dir = sys.argv[3]    # output save dir
TOPIC_NUM = int(sys.argv[4])  # topic number
HIDDEN_NUM = [500, 500] # hidden layer size
TOPIC_EMB_DIM = 150 # topic memory size
MAX_SEQ_LEN = 24    # clip length for a text
BATCH_SIZE = 32
MAX_EPOCH = 800
MIN_EPOCH = 50
PATIENT = 10
PATIENT_GLOBAL = 60
PRE_TRAIN_EPOCHS = 50
ALTER_TRAIN_EPOCHS = 50
TARGET_SPARSITY = 0.75
KL_GROWING_EPOCH = 0
SHORTCUT = True
TRANSFORM = None    # 'relu'|'softmax'|'tanh'
######################## configurations ########################

dataSeqTrain_fn = os.path.join(data_dir, "dataMsgTrain")
dataSeqTest_fn = os.path.join(data_dir, "dataMsgTest")
dataBowTrain_fn = os.path.join(data_dir, "dataMsgBowTrain")
dataBowTest_fn = os.path.join(data_dir, "dataMsgBowTest")
dataLabelTrain_fn = os.path.join(data_dir, "dataMsgLabelTrain")
dataLabelTest_fn = os.path.join(data_dir, "dataMsgLabelTest")

dataDictBow_fn = os.path.join(data_dir, "dataDictBow")
dictionary_bow = gensim.corpora.Dictionary.load(dataDictBow_fn)

dataDictSeq_fn = os.path.join(data_dir, "dataDictSeq")
dictionary_seq = gensim.corpora.Dictionary.load(dataDictSeq_fn)

# something need to save
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
sessLog_fn = os.path.join(output_dir, "session_%s.log" % datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
docTopicTrain_fn = os.path.join(output_dir, "doc_topic_train")
docTopicTest_fn = os.path.join(output_dir, "doc_topic_test")
topicWord_fn = os.path.join(output_dir, "topic_word")
topicWordSample_fn = os.path.join(output_dir, "topwords.txt")

bow_train = pickle.load(open(dataBowTrain_fn, 'rb'))
bow_test = pickle.load(open(dataBowTest_fn, 'rb'))
seq_train = pickle.load(open(dataSeqTrain_fn, 'rb'))
seq_test = pickle.load(open(dataSeqTest_fn, 'rb'))
label_train = pickle.load(open(dataLabelTrain_fn, 'rb'))
label_test = pickle.load(open(dataLabelTest_fn, 'rb'))

label_dict = json.load(open(os.path.join(data_dir, "labelDict.json")))
CATEGORY = len(label_dict)

def log(logfile, text, write_to_log=True):
    print(text)
    if write_to_log:
        with open(logfile, 'a') as f:
            f.write(text + '\n')

class CustomizedL1L2(regularizers.L1L2):
    def __init__(self, l1=0., l2=0.):
        self.l1 = K.variable(K.cast_to_floatx(l1))
        self.l2 = K.variable(K.cast_to_floatx(l2))

def generate_arrays_from_source(sp_mat):
    arrays = np.array(list(map(lambda x: np.squeeze(np.asarray(x.todense())), sp_mat)))
    index_arrays = np.zeros_like(arrays, dtype="int32")
    index_arrays[arrays > 0] = 1
    return normalize(arrays), index_arrays


def sampling(args):
    mu, log_sigma = args
    epsilon = K.random_normal(shape=(TOPIC_NUM,), mean=0.0, stddev=1.0)
    return mu + K.exp(log_sigma / 2) * epsilon


def print_weight_shape(model):
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()

    for name, weight in zip(names, weights):
        print(name, weight.shape)

def print_top_words(model, n=10):
    beta_exp = np.exp(model.get_weights()[-2])
    beta = beta_exp / (np.sum(beta_exp, 1)[:, np.newaxis])
    for k, beta_k in enumerate(beta):
        topic_words = [dictionary_bow[w_id] for w_id in np.argsort(beta_k)[:-n-1:-1]]
        print('Topic {}: {}'.format(k, ' '.join(topic_words)))

def output_theta(model, bow_input, fn):
    theta, _ = model.predict(bow_input)
    print("theta shape", theta.shape)
    pickle.dump(theta, open(fn, 'wb'))

def output_beta(model):
    beta_exp = np.exp(model.get_weights()[-2])
    beta = beta_exp / (np.sum(beta_exp, 1)[:, np.newaxis])
    pickle.dump(beta, open(topicWord_fn, 'wb'))
    with open(topicWordSample_fn, 'w') as fout:
        for k, beta_k in enumerate(beta):
            topic_words = [dictionary_bow[w_id] for w_id in np.argsort(beta_k)[:-11:-1]]
            fout.write("%s\n" % ' '.join(topic_words))


def check_sparsity(model, sparsity_threshold=1e-3):
    kernel = model.get_weights()[-2]
    num_weights = kernel.shape[0] * kernel.shape[1]
    num_zero = np.array(np.abs(kernel) < sparsity_threshold, dtype=float).sum()
    return num_zero / float(num_weights)


def update_l1(cur_l1, cur_sparsity, sparsity_target):
    current_l1 = K.get_value(cur_l1.l1)
    diff = sparsity_target - cur_sparsity
    new_l1 = current_l1 * 2.0 ** diff
    K.set_value(cur_l1.l1, K.cast_to_floatx(new_l1))

"""
main program
"""
# process input
seq_train_pad = pad_sequences(seq_train, maxlen=MAX_SEQ_LEN)
seq_test_pad = pad_sequences(seq_test, maxlen=MAX_SEQ_LEN)
label_train = keras.utils.to_categorical(label_train)
label_test = keras.utils.to_categorical(label_test)

bow_train, bow_train_ind = generate_arrays_from_source(bow_train)
bow_test, bow_test_ind = generate_arrays_from_source(bow_test)
test_count_indices = np.sum(bow_test_ind, axis=1)

# build model
bow_input = Input(shape=(len(dictionary_bow),), name="bow_input")     # the normalised input
seq_input = Input(shape=(MAX_SEQ_LEN, ), dtype='int32', name='seq_input')
embedding_mat = utils.build_embedding(embedding_fn, dictionary_seq, data_dir)
emb_dim = embedding_mat.shape[1]
seq_emb = Embedding(len(dictionary_seq) + 1,
                    emb_dim,
                    weights=[embedding_mat],
                    input_length=MAX_SEQ_LEN,
                    trainable=False)
topic_emb = Embedding(TOPIC_NUM, len(dictionary_bow), input_length=TOPIC_NUM, trainable=False)
psudo_input = Input(shape=(TOPIC_NUM, ), dtype='int32', name="psudo_input")

######################## build ntm #########################
# build encoder
e1 = Dense(HIDDEN_NUM[0], activation='relu')
e2 = Dense(HIDDEN_NUM[1], activation='relu')
e3 = Dense(TOPIC_NUM)
e4 = Dense(TOPIC_NUM)
h = e1(bow_input)
h = e2(h)
if SHORTCUT:
    es = Dense(HIDDEN_NUM[1], use_bias=False)
    h = add([h, es(bow_input)])

z_mean = e3(h)
z_log_var = e4(h)
# sample
hidden = Lambda(sampling, output_shape=(TOPIC_NUM,))([z_mean, z_log_var])
# build generator
g1 = Dense(TOPIC_NUM, activation="tanh")
g2 = Dense(TOPIC_NUM, activation="tanh")
g3 = Dense(TOPIC_NUM, activation="tanh")
g4 = Dense(TOPIC_NUM)

def generate(h):
    tmp = g1(h)
    tmp = g2(tmp)
    tmp = g3(tmp)
    tmp = g4(tmp)
    if SHORTCUT:
        r = add([Activation("tanh")(tmp), h])
    else:
        r = tmp
    if TRANSFORM is not None:
        r = Activation(TRANSFORM)(r)
        return r
    else:
        return r

represent = generate(hidden)
represent_mu = generate(z_mean)

# build decoder
l1_strength = CustomizedL1L2(l1=0.001)
d1 = Dense(len(dictionary_bow), activation="softmax", kernel_regularizer=l1_strength, name="p_x_given_h")
p_x_given_h = d1(represent)

# build classifier
filter_sizes = [1, 2, 3]
num_filters = 512
c1 = Dense(TOPIC_EMB_DIM, activation='relu')
t1 = Dense(TOPIC_EMB_DIM, activation='relu')
f1 = Dense(TOPIC_EMB_DIM, activation="relu")
f2 = Dense(TOPIC_EMB_DIM, activation="relu")
f3 = Dense(TOPIC_EMB_DIM, activation="relu")
f4 = Dense(TOPIC_EMB_DIM, activation="relu")
f5 = Dense(TOPIC_EMB_DIM, activation="relu")
o1 = Dense(TOPIC_EMB_DIM, activation='relu')
o2 = Dense(TOPIC_EMB_DIM, activation='relu')
o3 = Dense(TOPIC_EMB_DIM, activation='relu')
o4 = Dense(TOPIC_EMB_DIM, activation='relu')
o5 = Dense(TOPIC_EMB_DIM, activation='relu')

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], TOPIC_EMB_DIM), padding="valid",
                    kernel_initializer='normal', activation='relu')
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], TOPIC_EMB_DIM), padding="valid",
                kernel_initializer='normal', activation='relu')
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], TOPIC_EMB_DIM), padding="valid",
                kernel_initializer='normal', activation='relu')
s1 = Bidirectional(LSTM(80))
s2 = Dense(CATEGORY, activation='softmax')
cls_vars = [c1, t1, f1, o1, s1, s2]
x = seq_emb(seq_input)
x = c1(x)   # reducing dim
x = Dropout(0.05)(x)
wt_emb = topic_emb(psudo_input)
wt_emb = t1(wt_emb)     # reducing dim
# first match layer
match = dot([x, wt_emb], axes=(2, 2))
joint_match = add([represent_mu, match])
joint_match = f1(joint_match)
topic_sum = add([joint_match, x])
topic_sum = o1(topic_sum)
# # second match layer
# match = dot([topic_sum, wt_emb], axes=(2, 2))
# joint_match = add([represent_mu, match])
# joint_match = f2(joint_match)
# topic_sum = add([joint_match, x])
# topic_sum = o2(topic_sum)
# # third match layer
# match = dot([topic_sum, wt_emb], axes=(2, 2))
# joint_match = add([represent_mu, match])
# joint_match = f3(joint_match)
# topic_sum = add([joint_match, x])
# topic_sum = o3(topic_sum)
# # fourth match layer
# match = dot([topic_sum, wt_emb], axes=(2, 2))
# joint_match = add([represent_mu, match])
# joint_match = f4(joint_match)
# topic_sum = add([joint_match, x])
# topic_sum = o4(topic_sum)
# # fifth match layer
# match = dot([topic_sum, wt_emb], axes=(2, 2))
# joint_match = add([represent_mu, match])
# joint_match = f5(joint_match)
# topic_sum = add([joint_match, x])
# topic_sum = o5(topic_sum)

x = Reshape((MAX_SEQ_LEN, TOPIC_EMB_DIM, 1))(topic_sum)
x0 = conv_0(x)
x1 = conv_1(x)
x2 = conv_2(x)
mp0 = MaxPool2D(pool_size=(MAX_SEQ_LEN - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(x0)
mp1 = MaxPool2D(pool_size=(MAX_SEQ_LEN - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(x1)
mp2 = MaxPool2D(pool_size=(MAX_SEQ_LEN - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(x2)
out = concatenate([mp0, mp1, mp2], axis=1)
out = Dropout(0.05)(Flatten()(out))
cls_out = s2(out)

def kl_loss(x_true, x_decoded):
    kl_term = - 0.5 * K.sum(
        1 - K.square(z_mean) + z_log_var - K.exp(z_log_var),
        axis=-1)
    return kl_term

def nnl_loss(x_true, x_decoder):
    nnl_term = - K.sum(x_true * K.log(x_decoder + 1e-32), axis=-1)
    return nnl_term


kl_strength = K.variable(1.0)

# build combined model
ntm_model = Model(bow_input, [represent_mu, p_x_given_h])
ntm_model.compile(loss=[kl_loss, nnl_loss], loss_weights=[kl_strength, 1.0], optimizer="adagrad")

combine_model = Model([bow_input, seq_input, psudo_input], cls_out)
combine_model.compile(optimizer="adadelta", loss=K.categorical_crossentropy, metrics=["accuracy"])

vis_model = Model([bow_input, seq_input, psudo_input], [represent_mu, wt_emb, match, cls_out])

print_weight_shape(combine_model)

# init kl strength
num_batches = int(bow_train.shape[0] / BATCH_SIZE)
kl_base = float(KL_GROWING_EPOCH * num_batches)

optimize_ntm = True
first_optimize_ntm = True
min_bound_ntm = np.inf
min_bound_cls = - np.inf
epoch_since_improvement = 0
epoch_since_improvement_global = 0
# training
for epoch in range(1, MAX_EPOCH + 1):
    progress_bar = Progbar(target=num_batches)
    epoch_train = []
    epoch_test = []

    # shuffle data
    indices = np.arange(bow_train.shape[0])
    np.random.shuffle(indices)
    seq_train_shuffle = seq_train_pad[indices]
    bow_train_shuffle = bow_train[indices]
    bow_train_ind_shuffle = bow_train_ind[indices]
    label_train_shuffle = label_train[indices]
    psudo_indices = np.expand_dims(np.arange(TOPIC_NUM), axis=0)
    psudo_train = np.repeat(psudo_indices, seq_train_pad.shape[0], axis=0)
    psudo_test = np.repeat(psudo_indices, seq_test_pad.shape[0], axis=0)

    if optimize_ntm:
        print('Epoch {}/{} training {}'.format(epoch, MAX_EPOCH, "ntm"))
        for index in range(num_batches):
            # update kl_strength
            if epoch < KL_GROWING_EPOCH:
                K.set_value(kl_strength, np.float32((epoch * num_batches + index) / kl_base))
            else:
                K.set_value(kl_strength, 1.)
            bow_batch = bow_train_shuffle[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            bow_index_batch = bow_train_ind_shuffle[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            epoch_train.append(ntm_model.train_on_batch(
                bow_batch, [np.zeros([len(bow_batch), TOPIC_NUM]), bow_index_batch]))
            progress_bar.update(index + 1)
        # compute training cost

        [train_loss, train_kld, train_nnl] = np.mean(epoch_train, axis=0)
        print("ntm train loss: %.4f, kld: %.4f, nnl: %.4f" % (train_loss, train_kld, train_nnl))
        print_top_words(ntm_model)
        # check sparsity
        sparsity = check_sparsity(ntm_model)
        update_l1(l1_strength, sparsity, TARGET_SPARSITY)
        # estimate perplexity
        for j in range(5):
            epoch_test.append(ntm_model.evaluate(bow_test, [bow_test, bow_test_ind]))
        [val_loss, kld, nnl] = np.mean(epoch_test, axis=0)
        bound = np.exp(val_loss / np.mean(test_count_indices))
        print("ntm estimated perplexity upper bound on validation set: %.3f" % bound)
        # It is approximated perplexity
        # record the best perplexity
        if bound < min_bound_ntm and epoch > KL_GROWING_EPOCH:
            log(sessLog_fn, "New best val bound: %.3f in %d epoch" % (bound, epoch))
            min_bound_ntm = bound
            if first_optimize_ntm:
                print("Saving model")
                # ntm_model.save(Model_fn)
                output_theta(ntm_model, bow_train, docTopicTrain_fn)
                output_theta(ntm_model, bow_test, docTopicTest_fn)
            output_beta(ntm_model)
            epoch_since_improvement = 0
            epoch_since_improvement_global = 0
        elif bound >= min_bound_ntm:
            epoch_since_improvement += 1
            epoch_since_improvement_global += 1
            print("No improvement in epoch %d" % epoch)
        if epoch < KL_GROWING_EPOCH:
            print("Growing kl strength %.3f" % K.get_value(kl_strength))
        if epoch_since_improvement > PATIENT and epoch > MIN_EPOCH:
            optimize_ntm = False
            first_optimize_ntm = False
            epoch_since_improvement = 0
            beta_exp = np.exp(ntm_model.get_weights()[-2])
            beta = beta_exp / (np.sum(beta_exp, 1)[:, np.newaxis])
            topic_emb.set_weights([beta])   # update topic-word matrix
            # min_bound_ntm += 2    # relax ntm bound a bit
        if epoch_since_improvement_global > PATIENT_GLOBAL:
            break
    else:
        print('Epoch {}/{} training {}'.format(epoch, MAX_EPOCH, "cls"))
        for index in range(num_batches):
            bow_batch = bow_train_shuffle[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            seq_batch = seq_train_shuffle[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            psudo_batch = psudo_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            label_batch = label_train_shuffle[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            epoch_train.append(combine_model.train_on_batch(
                [bow_batch, seq_batch, psudo_batch], label_batch))
            progress_bar.update(index + 1)
        train_loss, train_acc = np.mean(epoch_train, axis=0)
        print("cls train loss: %.4f" % (train_loss))
        y_pred = combine_model.predict([bow_test, seq_test_pad, psudo_test])
        y_pred_label = np.argmax(y_pred, axis=1)
        y_true_label = np.argmax(label_test, axis=1)
        test_acc = accuracy_score(y_true_label, y_pred_label)
        test_f1 = f1_score(y_true_label, y_pred_label, average="weighted")
        if test_acc > min_bound_cls:
            min_bound_cls = test_acc
            log(sessLog_fn, "New best val acc: %.4f, f1: %.4f in %d epoch" % (min_bound_cls, test_f1, epoch))
            epoch_since_improvement = 0
            epoch_since_improvement_global = 0

        else:
            epoch_since_improvement += 1
            epoch_since_improvement_global += 1
            print("No improvement in epoch %d with val acc %.4f, f1 %.4f" % (epoch, test_acc, test_f1))

        if epoch_since_improvement > PATIENT and epoch > MIN_EPOCH:
            optimize_ntm = True
            epoch_since_improvement = 0
        if epoch_since_improvement_global > PATIENT_GLOBAL:
            break
