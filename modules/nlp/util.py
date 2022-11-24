# setting pretty pandas dataframe show
import csv
import os
import re

import torch
import gensim
import nltk
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoTokenizer, AutoModel

nltk.download("stopwords")
from nltk.corpus import stopwords
from pymystem3 import Mystem


def prettify_pandas_print():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth', None)
    pd.set_option('max_seq_item', None)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    pd.set_option('precision', 10)


def get_default_paths(root):
    root_data = f"{root}/data"
    root_images = f"{root}/data/images"
    root_models = f"{root}/models"
    root_graphs = f"{root}/graphs"
    if not os.path.exists(root):
        os.mkdir(root)

    if not os.path.exists(root_data):
        os.mkdir(root_data)

    if not os.path.exists(root_images):
        os.mkdir(root_images)

    if not os.path.exists(root_models):
        os.mkdir(root_models)

    if not os.path.exists(root_graphs):
        os.mkdir(root_graphs)
    return root, root_data, root_models, root_graphs, root_images


def ohe_col_with_threshold(df, col, threshold):
    df[col] = df[col].astype(str).fillna('N/A').str.lower()
    df.loc[df[col].value_counts()[df[col]].values < threshold, col] = "rare"
    df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
    return df


# making ohe for columns replacing values lower threshold with rare
def ohe_with_threshold(df, columns, threshold):
    for col in columns:
        df = ohe_col_with_threshold(df, col, threshold)
    return df


# bounding value in [l_threshold, r_threshold] and making log scaling
def log_scale(df, col, l_threshold, r_threshold):
    df[col] = np.log1p(df[col].astype(float).clip(l_threshold, r_threshold) + 1e-5)
    return df


# fill empty column values with val
def fill_value(df, col, val=0):
    df[col] = df[col].fillna(val)
    return df


# fill empty column values with mean
def fill_mean(df, col):
    df[col] = df[col].astype(float)
    mean_value = df[col].mean()
    df[col] = df[col].fillna(mean_value)
    return df


def add_mean_by(df, col, by):
    if type(by) is not list:
        by = [by]
    agg_vals = df[[col] + by].groupby(by)[col].describe()
    mean_df = agg_vals['mean'].reset_index()
    mean_df.columns = by + ['{}_mean'.format(col)]
    df = df.merge(mean_df)
    return df


def add_std_by(df, col, by):
    if type(by) is not list:
        by = [by]
    agg_vals = df[[col] + by].groupby(by)[col].describe()
    std_df = agg_vals['std'].reset_index()
    std_df.columns = by + ['{}_std'.format(col)]
    df = df.merge(std_df)
    return df


# fill empty column values by mean grouping with by
def fill_mean_by(df, col, by, drop_mean=True):
    mean_df = add_mean_by(df, col, by)
    df = df.merge(mean_df)
    df[col] = df[col].fillna(df['{}_mean'.format(col)])
    if drop_mean:
        df = df.drop('{}_mean'.format(col), axis=1)
    return df


# create batches with size sz from python list
def create_batches(lst, sz):
    return [lst[i:i + sz] for i in range(0, len(lst), sz)]


# split dataframe to chunks
def create_df_chunks(df, num):
    return np.array_split(df, num)


# applies function func to pandas chunks
def apply_chunks(df_chunks, func):
    for i, chunk in enumerate(df_chunks):
        func(chunk, i)


# process russian column with russian stopwords from nltk and pymystem3 from yandex
def process_russian_col(df, col):
    nltk.download('stopwords')
    text_batch_size = 1000
    df[col] = df[col].fillna(' ')
    series = df[col].tolist()
    batches = create_batches(series, text_batch_size)

    russian_stopwords = stopwords.words("russian")
    mystem = Mystem()

    result = []
    for txt_part in batches:
        concated = ' '.join([re.sub('([^а-я]+)', ' ', str(txt).lower()) + ' brk ' for txt in txt_part])
        words = mystem.lemmatize(concated)
        sen = []
        for word in words:
            word = word.strip()
            if word != '\n' and word != '' and word not in russian_stopwords:
                if word == 'brk':
                    result.append(' '.join(sen))
                    sen = []
                else:
                    sen.append(word)

    df[col] = pd.Series(result)

    return df


def process_russian_cols(df, cols):
    for col in cols:
        df = process_russian_col(df, col)
    return df


def save_text_col(df, i, col, path, fillna):
    df = df[col].fillna(fillna)
    df.to_csv(path, header=False, index=False, sep=' ', encoding='utf-8',
              quoting=csv.QUOTE_NONE, escapechar=' ', mode='w' if i == 0 else 'a')


# build doc2vec model from text and save it to {tmp_path}/doc2vec_{col}
def build_doc2vec(df, col, chunked, fill_na, tmp_path, vector_size, window_size, epochs, workers=2, min_word_count=10):
    txt_path = '{}/{}.txt'.format(tmp_path, col)

    if chunked:
        apply_chunks(df_chunks=df, func=lambda chunk, i: save_text_col(chunk, i, col, txt_path, fill_na))
    else:
        save_text_col(df, 0, col, txt_path, fill_na)

    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size,
                                          window=window_size,
                                          min_count=min_word_count, epochs=epochs, workers=workers)

    model.build_vocab(corpus_file=txt_path)
    model.train(corpus_file=txt_path,
                total_examples=model.corpus_count, epochs=model.epochs, total_words=model.corpus_count)
    model.save('{}/doc2vec_{}'.format(tmp_path, col))

    os.remove(txt_path)


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def get_tiny_bert_model(gpu=False):
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
    model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
    if gpu:
        model.cuda()
    return model, tokenizer


def embed_bert_cls(text, tokenizer, model, torch):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output[-2][:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()


def add_tiny_bert_embeds(df, col, fillna, model, tokenizer, pref="bert", gpu=False):
    df[col] = df[col].astype(str).fillna(fillna)
    embeds = [embed_bert_cls(str(text), tokenizer, model, torch) for text in df[col].tolist()]
    df = pd.concat([df, pd.DataFrame(embeds,
                                     columns=["{}_{}_{}".format(pref, col, i) for i in range(len(embeds[0]))]).astype(
        float)],
                   axis=1)
    return df


def add_doc2vec_embeds(df, col, fillna, model_path, pref="doc2vec"):
    model = gensim.models.Doc2Vec.load(model_path)
    embeds = [model.infer_vector(gensim.utils.simple_preprocess(str(text))) for text in
              df[col].astype(str).fillna(fillna).tolist()]
    df = pd.concat([df, pd.DataFrame(embeds,
                                     columns=["{}_{}_{}".format(pref, col, i) for i in range(len(embeds[0]))]).astype(
        float)],
                   axis=1)
    return df


def create_empty_image(h, w, path):
    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(path)


def __process_image_path(file_path, h, w, tf):
    img = tf.io.read_file(file_path)
    if tf.io.is_jpeg(img):
        img = tf.cast(tf.io.decode_jpeg(img, channels=3), tf.uint8)
    else:
        img = tf.cast(tf.zeros((h, w, 3)), tf.uint8)
    img = tf.image.resize(img, [h, w])
    return img


def get_items_starts_with(items, lst):
    return [val for val in items if val.startswith(tuple(lst))]
