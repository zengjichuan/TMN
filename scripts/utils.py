from keras import backend as K
import numpy as np
import os

def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)

def build_embedding(embedding_fn, dictionary, data_dir):
    print("building embedding matrix for dict %d if need..." % len(dictionary))
    embedding_mat_fn = os.path.join(data_dir, "embedding_mat_%d.npy" % (len(dictionary)))
    if os.path.exists(embedding_mat_fn):
        embedding_mat = np.load(embedding_mat_fn)
        return embedding_mat
    embedding_index = {}
    with open(embedding_fn) as fin:
        first_line = True
        l_id = 0
        for line in fin:
            if l_id % 100000 == 0:
                print("loaded %d words embedding..." % l_id)
            if ("glove" not in embedding_fn) and first_line:
                first_line = False
                continue
            line = line.rstrip()
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
            l_id += 1
    embedding_dim = len(embedding_index.values()[0])
    embedding_mat = np.zeros((len(dictionary) + 1, embedding_dim))    # 0 is for padding
    for i, word in dictionary.items():
        embedding_vec = embedding_index.get(word)
        if embedding_vec is not None:
            embedding_mat[i + 1] = embedding_vec
    np.save(embedding_mat_fn, embedding_mat)
    return embedding_mat


def unfreeze(layers):
    for layer in layers:
        layer.trainable = True


def freeze(layers):
    for layer in layers:
        layer.trainable = False
