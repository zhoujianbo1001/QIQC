# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.
from tqdm import tqdm
tqdm.pandas()
import re
import gc
import time

import string

from gensim.models import KeyedVectors

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import Layer
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D

from keras.models import Model
from keras.models import load_model

#################################### Params ##########################################
DATA_PATH = "../input"
# DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
GLOVE_PATH = os.path.join(DATA_PATH, "embeddings", "glove.840B.300d", "glove.840B.300d.txt")
glove_mean = -0.005838499
glove_std = 0.48782197
GOOGLENEWS_PATH = os.path.join(DATA_PATH, "embeddings", "GoogleNews-vectors-negative300", "GoogleNews-vectors-negative300.bin")
PARAGRAM_PATH =  os.path.join(DATA_PATH, "embeddings", "paragram_300_sl999", "paragram_300_sl999.txt")
paragram_mean = -0.0053247944
paragram_std = 0.49346468
WIKI_NEWS = os.path.join(DATA_PATH, "embeddings", "wiki-news-300d-1M", "wiki-news-300d-1M.vec")
print(os.listdir(DATA_PATH))

vocab_size = 120000
max_seq_len = 72
emb_size = 300
######################################################################################
# Precess data

# clean data
def clean_data(sentence):
    # 1. lower
    x = str(sentence)
    x = x.lower()
    
    # 2. punct data
    puncts = [
        ',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
        '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
        '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
        '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
        '∙', '）', '↓', '、', '│', '₹', 'π', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    
    # 3. number data
    x = re.sub('[0-9]{5,}', ' ##### ', x)
    x = re.sub('[0-9]{4}', ' #### ', x)
    x = re.sub('[0-9]{3}', ' ### ', x)
    x = re.sub('[0-9]{2}', ' ## ', x)
    
    # 4. mispell data
    mispell_dict = {
        "aren't" : "are not",
        "can't" : "cannot",
        "couldn't" : "could not",
        "didn't" : "did not",
        "doesn't" : "does not",
        "don't" : "do not",
        "hadn't" : "had not",
        "hasn't" : "has not",
        "haven't" : "have not",
        "he'd" : "he would",
        "he'll" : "he will",
        "he's" : "he is",
        "i'd" : "I would",
        "i'd" : "I had",
        "i'll" : "I will",
        "i'm" : "I am",
        "isn't" : "is not",
        "it's" : "it is",
        "it'll":"it will",
        "i've" : "I have",
        "let's" : "let us",
        "mightn't" : "might not",
        "mustn't" : "must not",
        "shan't" : "shall not",
        "she'd" : "she would",
        "she'll" : "she will",
        "she's" : "she is",
        "shouldn't" : "should not",
        "that's" : "that is",
        "there's" : "there is",
        "they'd" : "they would",
        "they'll" : "they will",
        "they're" : "they are",
        "they've" : "they have",
        "we'd" : "we would",
        "we're" : "we are",
        "weren't" : "were not",
        "we've" : "we have",
        "what'll" : "what will",
        "what're" : "what are",
        "what's" : "what is",
        "what've" : "what have",
        "where's" : "where is",
        "who'd" : "who would",
        "who'll" : "who will",
        "who're" : "who are",
        "who's" : "who is",
        "who've" : "who have",
        "won't" : "will not",
        "wouldn't" : "would not",
        "you'd" : "you would",
        "you'll" : "you will",
        "you're" : "you are",
        "you've" : "you have",
        "'re": " are",
        "wasn't": "was not",
        "we'll":" will",
        "didn't": "did not",
        "tryin'":"trying",

        'colour':'color',
        'centre':'center',
        'didnt':'did not',
        'doesnt':'does not',
        'isnt':'is not',
        'shouldnt':'should not',
        'favourite':'favorite',
        'travelling':'traveling',
        'counselling':'counseling',
        'theatre':'theater',
        'cancelled':'canceled',
        'labour':'labor',
        'organisation':'organization',
        'wwii':'world war 2',
        'citicise':'criticize',
        'instagram': 'social medium',
        'whatsapp': 'social medium',
        'snapchat': 'social medium'
    }
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    def replace(match):
        return mispell_dict[match.group(0)]
    x = mispell_re.sub(replace, x)
    
    
    # 5. stop words
    tmp = ""
    for word in x.split():
        if word not in list(STOP_WORDS):
            tmp = tmp + word + " "
    x = tmp
    
    return x

# read_data
def read_data():
    train_df = pd.read_csv(os.path.join(DATA_PATH,"train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

    # # Under sampling
    # insinceres_df = train_df.loc[train_df["target"] == 1]
    # sinceres_df = train_df.loc[train_df["target"] == 0].sample(n=len(insinceres_df))
    # train_df = pd.concat([insinceres_df, sinceres_df])

    print(train_df.head(5))
    print(test_df.head(5))

    # clean data
    train_df["question_text"] = train_df["question_text"].progress_apply(
        lambda x: clean_data(x))
    test_df["question_text"] = test_df["question_text"].progress_apply(
        lambda x: clean_data(x))
    print(train_df["question_text"].head(5))
    print(test_df["question_text"].head(5))

    # train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2019)
        
    print("Train shape : ", train_df.shape)
    print("Test shape : ", test_df.shape)

    # Fill NaN
    train_x = train_df["question_text"].fillna("_na_").values
    test_x = test_df["question_text"].fillna("_na_").values
    print(type(train_x))
    print(train_x[0])

    train_y = train_df['target'].values

    # Tokenize the sentences
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(list(train_x))
    print("word_index_len : ", len(tokenizer.word_index))

    train_x = tokenizer.texts_to_sequences(train_x)
    test_x = tokenizer.texts_to_sequences(test_x)
    print(type(train_x))
    print(train_x[0])

    # Pad the sentences 
    train_x = pad_sequences(train_x, maxlen=max_seq_len)
    test_x = pad_sequences(test_x, maxlen=max_seq_len)
    print(type(train_x))
    print(train_x[0])
    print(type(train_x[0]))

    return train_x, train_y, test_x, tokenizer.word_index

def load_embedding(file):
    def get_coefs(word, *arr): 
        return word, np.asarray(arr, dtype='float32')
    if file == GOOGLENEWS_PATH:
        embeddings_index = KeyedVectors.load_word2vec_format(file, binary=True)
    elif file == os.path.join(DATA_PATH, "embeddings", "wiki-news-300d-1M", "wiki-news-300d-1M.vec"):
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
        
    return embeddings_index

def build_vocab(sentences, verbose=True):
    # Tokenize the sentences
    tokenizer = Tokenizer(
        num_words=vocab_size,
        filters = '!"#$%()*+,-./:;<=>?@[\]^_`{|}~ '
    )
    tokenizer.fit_on_texts(sentences)
    vocab = tokenizer.word_index
    print("word_index_len : ", len(vocab))

    return vocab

# Make embedding matrixs
def make_embedding_matrixs(
    embedding_index_dict, 
    emb_mean, emb_std,
    word_index_dict, 
    vocab_size=vocab_size, emb_size=emb_size):
    # all_embs = np.stack(embedding_index_dict.values())
    # emb_mean, emb_std = all_embs.mean(), all_embs.std()
    # print("emb_mean : ", emb_mean)
    # print("emb_std : ", emb_std)
    # embed_size = all_embs.shape[1]

    nb_words = min(vocab_size, len(word_index_dict))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, emb_size))
    for word, i in word_index_dict.items():
        if i >= vocab_size: continue
        embedding_vector = embedding_index_dict.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

##########################################################################################
# Define the module
class Attention(Layer):
    def __init__(
        self, step_dim,
        W_regularizer=None, b_regularizer=None,
        W_constraint=None, b_constraint=None,
        bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(
            (input_shape[-1],),
            initializer=self.init,
            name='{}_W'.format(self.name),
            regularizer=self.W_regularizer,
            constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(
                (input_shape[1],),
                initializer='zero',
                name='{}_b'.format(self.name),
                regularizer=self.b_regularizer,
                constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(
            K.dot(
                K.reshape(x, (-1, features_dim)),
                K.reshape(self.W, (features_dim, 1))
                ), 
            (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

# Define the model
def model_cnn(embedding_matrix):
    filter_sizes = [1,2,3,5]
    num_filters = 36

    inp = Input(shape=(max_seq_len,))
    x = Embedding(vocab_size, emb_size, weights=[embedding_matrix])(inp)
    x = Reshape((max_seq_len, emb_size, 1))(x)

    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], emb_size),
                                     kernel_initializer='he_normal', activation='elu')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(max_seq_len - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool)   
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(1, activation="sigmoid")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def model_lstm_atten(embedding_matrix):
    inp = Input(shape=(max_seq_len,))
    x = Embedding(vocab_size, emb_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = Attention(max_seq_len)(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def model_lstm_du(embedding_matrix):
    inp = Input(shape=(max_seq_len,))
    x = Embedding(vocab_size, emb_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def model_gru_srk_atten(embedding_matrix):
    inp = Input(shape=(max_seq_len,))
    x = Embedding(vocab_size, emb_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = Attention(max_seq_len)(x) # New
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model   

def model_gru_atten_3(embedding_matrix):
    inp = Input(shape=(max_seq_len,))
    x = Embedding(vocab_size, emb_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(100, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = Attention(max_seq_len)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def QIQC_model(inputs):
    inp = Input(shape=(num_models,))
    x = Dense(1, activation="sigmoid")
    model = Model(inputs=inp, outputs=x)
    
    model.compile(
        loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Define train and predict function
def train_pred(
    train_x, train_y, 
    val_x, val_y, test_x,
    model, model_name, epochs=2):
    for e in range(epochs):
        model.fit(
            train_x, train_y, 
            batch_size=512, epochs=1, 
            validation_data=(val_x, val_y))
    # Save the model
    model.save("model.h5")

    # Load the model
    # model = load_model("model.h5")
    
    # Validation
    pred_val_y = model.predict([val_x], batch_size=1024, verbose=0)
    
    # Predict
    pred_test_y = model.predict([test_x], batch_size=1024, verbose=0)
    return pred_val_y, pred_test_y

# Define threshold search function
def threshold_search(y_true, y_pred):
    best_thresh = 0
    max_f1_score = 0.0
    for thresh in np.arange(0.1, 0.601, 0.01):
        thresh = np.round(thresh, 2)
        f1_score = metrics.f1_score(y_true, (y_pred>thresh).astype(int))
        print("F1 score at threshold {0} is {1}".format(thresh, f1_score))
        if max_f1_score < f1_score:
            max_f1_score = f1_score
            best_thresh = thresh
    return best_thresh, max_f1_score
    

##################################################################################################    
def kernel_main():
    # Train data
    start_time = time.time()
    # Read data
    print("Begin read data ...")
    train_x, train_y, test_x, word_index_dict = read_data()
    print("Reading data completed!")
    # Load glove embedding
    print("Begin load glove embedding ...")
    embedding_index_dict = load_embedding(GLOVE_PATH)
    print("Load glove data completed!")
    # Make embedding matrix
    print("Begin make embedding matrix ...")
    embedding_matrix_glove = make_embedding_matrixs(
        embedding_index_dict, glove_mean, glove_std, word_index_dict)
    print("Make embedding matrix completed! ")
    # # GC
    # del embedding_index_dict
    # gc.collect()
    # # Load paragram embedding
    # print("Begin load paragram embedding ...")
    # embedding_index_dict = load_embedding(PARAGRAM_PATH)
    # print("Load paragram data completed!")
    # # Make embedding matrix
    # print("Begin make embedding matrix ...")
    # embedding_matrix_paragram = make_embedding_matrixs(
    #     embedding_index_dict, paragram_mean, paragram_std, word_index_dict)
    # print("Make embedding matrix completed! ")
    # # GC
    # del embedding_index_dict
    # gc.collect()
    # # Compute embedding mean
    # print("Begin compute embedding mean ...")
    # embedding_matrix = np.mean([embedding_matrix_glove, embedding_matrix_paragram], axis=0)
    # print("Compute embedding mean completed!")
    # # GC
    # del word_index_dict
    # gc.collect()
    # # # Load Wiki_news embedding
    # # print("Begin load Wiki_news embedding ...")
    # # embedding_index_dict = load_embedding(WIKI_NEWS)
    # # print("Load Wiki_news embedding completed!")
    # # # Make embedding matrix
    # # print("Begin make embedding matrix ...")
    # # embedding_matrix_1 = make_embedding_matrixs(embedding_index_dict, word_index_dict)
    # # print("Make embedding matrix completed! ")
    # # # GC
    # # del embedding_index_dict, word_index_dict
    # # gc.collect()
    # # # Compute embedding mean
    # # print("Begin compute embedding mean ...")
    # # embedding_matrix = np.mean([embedding_matrix_1, embedding_matrix], axis=0)
    # # print("Compute embedding mean completed!")
    # # # GC
    # # del embedding_matrix_1
    # # gc.collect()

    # total_time = (time.time() - start_time) / 60
    # print("Reading data takes {:.2f} minutes".format(total_time))

    # outputs = []

    # pred_val_y, pred_test_y = train_pred(
    #     train_x, train_y , val_x, val_y, test_x,
    #     model_gru_atten_3(embedding_matrix), epochs = 3)
    # outputs.append([pred_val_y, pred_test_y, '3 GRU w/ atten'])

    # pred_val_y, pred_test_y = train_pred(
    #     train_x, train_y , val_x, val_y, test_x,
    #     model_gru_srk_atten(embedding_matrix), epochs = 2)
    # outputs.append([pred_val_y, pred_test_y, 'gru atten srk'])

    # pred_val_y, pred_test_y = train_pred(
    #     train_x, train_y , val_x, val_y, test_x,
    #     model_cnn(embedding_matrix_glove), epochs = 2) # GloVe only
    # outputs.append([pred_val_y, pred_test_y, '2d CNN GloVe'])

    # pred_val_y, pred_test_y = train_pred(
    #     train_x, train_y , val_x, val_y, test_x,
    #     model_lstm_du(embedding_matrix), epochs = 2)
    # outputs.append([pred_val_y, pred_test_y, 'LSTM DU'])

    # pred_val_y, pred_test_y = train_pred(
    #     train_x, train_y , val_x, val_y, test_x,
    #     model_lstm_atten(embedding_matrix), epochs = 3)
    # outputs.append([pred_val_y, pred_test_y, '2 LSTM w/ attention'])

    # pred_val_y, pred_test_y = train_pred(
    #     train_x, train_y , val_x, val_y, test_x,
    #     model_lstm_atten(embedding_matrix_glove), epochs = 3) # Only GloVe
    # outputs.append([pred_val_y, pred_test_y, '2 LSTM w/ attention GloVe'])

    # pred_val_y, pred_test_y = train_pred(
    #     train_x, train_y , val_x, val_y, test_x,
    #     model_lstm_atten(embedding_matrix_paragram), epochs = 3) # Only Para
    # outputs.append([pred_val_y, pred_test_y, '2 LSTM w/ attention Para'])

    # coefs = np.array([1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7], np.float32)
    # # coefs = [0.20076554,0.07993707,0.11611663,0.14885248,0.15734404,0.17454667,0.14288361]
    # pred_test_y = np.sum([outputs[i][1]*coefs[i] for i in range(len(coefs))], axis = 0)


    # best_thresh, max_f1_score = threshold_search(val_y, pred_val_y)
    # print("pre_thresh : ", best_thresh)
    # print("max_f1_score : ", max_f1_score)

    # # submission
    # pred_test_y = np.squeeze(pred_test_y)
    # test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
    # my_sub = pd.DataFrame({'qid':test_df['qid'], 'prediction':(pred_test_y>best_thresh).astype(int)})
    # my_sub.to_csv("submission.csv", index=False)
    # print("Completed!")

if __name__ == "__main__":
    kernel_main()