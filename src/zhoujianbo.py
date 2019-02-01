import os

import pandas as pd
import numpy as np

from tqdm import tqdm
tqdm.pandas()
import re
import gc
import time

import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from gensim.models import KeyedVectors

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

import sklearn as sl
from sklearn.model_selection import train_test_split


from transformer_encoder import *

############################ Model #######################################

class TransformerModel(tf.keras.Model):
    def __init__(self, embedding_matrix):
        super(TransformerModel, self).__init__()
        self.embedding_matrix = embedding_matrix
        self.dense1 = tf.layers.Dense(units=32, activation="relu")
        self.dense2 = tf.layers.Dense(units=1, activation="sigmoid")
        
    
    def call(
        self, 
        inputs, 
        training=False):
        # [batch_size, seq_len, emb_dim]
        x = tf.nn.embedding_lookup(self.embedding_matrix, inputs)
        print(x)

        # [batch_size, seq_len, output_size]
        x = transformer_encoder(x, 64)

        # pooling
        # [batch_size, output_size]
        max_x = tf.reduce_max(x, axis=1)
        # mean_x = tf.reduce_mean(x, axis=1)
        print(max_x)
        # concat
        # x = tf.concat([max_x, mean_x], axis=-1)

        # forward
        # [batch_size, out_size]
        x = self.dense1(max_x)
        print(x)
        logits = self.dense2(x)
        print(logits)

        return logits


#################################### Params ##########################################
# DATA_PATH = "../input"
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input")
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

######################################## process data ##########################################
# clean data
def clean_data(sentence):
    x = str(sentence)
    
    # punct data
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
    
    # number data
    x = re.sub('[0-9]{5,}', ' ##### ', x)
    x = re.sub('[0-9]{4}', ' #### ', x)
    x = re.sub('[0-9]{3}', ' ### ', x)
    x = re.sub('[0-9]{2}', ' ## ', x)
    
    # mispell data
    mispell_dict = {
        "ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'
        }
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    def replace(match):
        return mispell_dict[match.group(0)]
    x = mispell_re.sub(replace, x)
    
    
    # stop words
    tmp = ""
    for word in x.split():
        if word not in list(STOP_WORDS):
            tmp = tmp + word + " "
    x = tmp
    
    # lower words
    x = x.lower()

    return x

# read data
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

# def build_vocab(sentences, verbose=True):
#     # Tokenize the sentences
#     tokenizer = Tokenizer(
#         num_words=vocab_size,
#         filters = '!"#$%()*+,-./:;<=>?@[\]^_`{|}~ '
#     )
#     tokenizer.fit_on_texts(sentences)
#     vocab = tokenizer.word_index
#     print("word_index_len : ", len(vocab))

#     return vocab

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

# save process data
def save_process_data():
    # Train data
    start_time = time.time()
    # Read data
    print("Begin read data ...")
    train_x, train_y, test_x, word_index_dict = read_data()
    print("Reading data completed!")
    np.save("train_x",train_x)
    np.save("train_y",train_y)
    np.save("test_x",test_x)

    np.save("word_index.npy",word_index_dict)

    # Load glove embedding
    print("Begin load glove embedding ...")
    embedding_index_dict = load_embedding(GLOVE_PATH)
    print("Load glove data completed!")
    # Make embedding matrix
    print("Begin make embedding matrix ...")
    embedding_matrix_glove = make_embedding_matrixs(
        embedding_index_dict, glove_mean, glove_std, word_index_dict)
    print("Make embedding matrix completed! ")
    # GC
    del embedding_index_dict
    gc.collect()
    # Load paragram embedding
    print("Begin load paragram embedding ...")
    embedding_index_dict = load_embedding(PARAGRAM_PATH)
    print("Load paragram data completed!")
    # Make embedding matrix
    print("Begin make embedding matrix ...")
    embedding_matrix_paragram = make_embedding_matrixs(
        embedding_index_dict, paragram_mean, paragram_std, word_index_dict)
    print("Make embedding matrix completed! ")
    # GC
    del embedding_index_dict
    gc.collect()
    # Compute embedding mean
    print("Begin compute embedding mean ...")
    embedding_matrix = np.mean([embedding_matrix_glove, embedding_matrix_paragram], axis=0)
    print("Compute embedding mean completed!")
    np.save("embedding_matrix", embedding_matrix)

    total_time = (time.time() - start_time) / 60
    print("Reading data takes {:.2f} minutes".format(total_time))

######################################################################################################

################################### train data ####################################################
lr = 1e-3
EPOCH = 3
BATCH_SIZE = 512

def train():
    # load data
    train_x = np.load("train_x.npy")
    train_y = np.load("train_y.npy")
    test_x = np.load("test_x.npy")
    embedding_matrix = np.load("embedding_matrix.npy")
    embedding_matrix = embedding_matrix.astype(np.float32)
    print(train_x.shape)
    print(train_y.shape)
    # define model
    inputs = tf.keras.Input(
        shape=(max_seq_len, ), 
        dtype=np.int32)
    myModel = TransformerModel(embedding_matrix)
    myModel.apply(inputs)
    myModel.compile(
        loss='binary_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
    )
    print(myModel.summary())
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.keras.backend.set_session(sess)
    myModel.fit(
        train_x, train_y, 
        batch_size=BATCH_SIZE, epochs=3,
        verbose=1, validation_split=0.1
    )

    myModel.predict([test_x], batch_size=512, verbose=1)
    
    sess.close()



if __name__ == "__main__":
    # save_process_data()
    train()