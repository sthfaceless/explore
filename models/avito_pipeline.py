import gc
import glob
import pandas as pd
import tensorflow as tf
import tabnet
import numpy as np
import gensim

from models.utils import *

# paths
# ROOT, ROOT_DATA, ROOT_MODELS, ROOT_GRAPHS, ROOT_IMAGES = get_default_paths('/content/drive/MyDrive/ml/hacks/avito')
ROOT, ROOT_DATA, ROOT_MODELS, ROOT_GRAPHS, ROOT_IMAGES = get_default_paths('/home/danil/PycharmProjects/explore')

# images
IMG_HEIGHT, IMG_WIDTH = 224, 224
image_batch_size = 100
model_link = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"
image_prefix = "image_"

# text
title_batch_size = 100
title_vector_size = 100
title_window_size = 5
description_batch_size = 100
description_vector_size = 300
description_window_size = 10
bert_prefix = "bert_"
doc2vec_prefix = "doc2vec_"
doc2vec_workers = 8
doc2vec_epochs = 30

# tabnet
tabnet_batch_size = 1000
tabnet_shuffle_size = 10000
tabnet_epochs = 5
tabnet_lr = 1e-4
tabnet_layers = 5
tabnet_steps_epoch = 100
tabnet_valid_steps = 100
tabnet_feature_relaxation = 1.2
tabnet_sparsity_coeff = 1e-3
tabnet_batch_momentum = 0.8
tabnet_virtual_batch_size = 250
tabnet_feature_dim = 64
tabnet_output_dim = 32

# general
train_size = int(1000)
valid_size = train_size * 0.3
batch_size = 100
categorical_threshold = 10

# columns
categorical_columns = ['region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3',
                       'user_type', 'item_seq_number']
gen_categorical_columns = [col + "_" for col in categorical_columns]
numerical_columns = ['image_top_1', 'price']
gen_numerical_columns = ['price_mean', 'price_std']
text_columns = ['title', 'description']
meta_columns = ['item_id', 'user_id', 'image', 'activation_date']
label = 'deal_probability'
feature_columns = numerical_columns + gen_numerical_columns + gen_categorical_columns \
                  + [bert_prefix, doc2vec_prefix, image_prefix]

# set pretty output for pandas dataframes
prettify_pandas_print()


def make_features():
    # load train data and sort it for avoiding future leak
    train = pd.read_csv(f'{ROOT_DATA}/train.csv').sort_values('activation_date')
    train['is_train'] = '1'

    test = pd.read_csv(f'{ROOT_DATA}/test.csv')
    test['is_train'] = '0'

    df = pd.concat([train, test], axis=0)

    # price has lognormal distribution
    df = log_scale(df, 'price', 0, 1e6)
    df = fill_mean_by(df, 'price', 'parent_category_name', drop_mean=False)
    df = add_std_by(df, 'price', 'parent_category_name')
    df = fill_value(df, 'image_top_1', 0)

    # clean and lemmatize russian text
    df = process_russian_cols(df, text_columns)

    # create ohe for category
    for col in categorical_columns:
        df = ohe_col_with_threshold(df, col, categorical_threshold)

    df[df['is_train'] == '1'].to_csv(f'{ROOT_DATA}/train_features.csv', index=False, header=True)
    df[df['is_train'] == '0'].to_csv(f'{ROOT_DATA}/test_features.csv', index=False, header=True)

    del df
    gc.collect()


# making base features for train/test
make_features()


# preparing for text
def get_chunks(name):
    return pd.read_csv(f'{ROOT_DATA}/{name}_features.csv', chunksize=batch_size)


build_doc2vec(df=get_chunks('train'), col='title', chunked=True, fill_na='продать', workers=doc2vec_workers,
              tmp_path=ROOT_MODELS, vector_size=title_vector_size, window_size=title_window_size, epochs=doc2vec_epochs)
build_doc2vec(df=get_chunks('train'), col='description', chunked=True, fill_na='продать', workers=doc2vec_workers,
              tmp_path=ROOT_MODELS, vector_size=description_vector_size, window_size=description_window_size,
              epochs=doc2vec_epochs)

bert_model, bert_tokenizer = get_tiny_bert_model()
#
# # preparing for images
create_empty_image(IMG_WIDTH, IMG_HEIGHT, f"{ROOT_IMAGES}/empty.jpg")
image_paths = {image_path.split("/")[-1].replace(".jpg", ""): image_path for image_path in
               glob.glob(f"{ROOT_IMAGES}/*.jpg")}
image_model = create_image_model(IMG_HEIGHT, IMG_WIDTH, model_link)


def add_embeddings(name):
    first_valid = True
    for i, chunk in enumerate(get_chunks(name)):
        chunk = add_tiny_bert_embeds(chunk, 'title', fillna='продать', model=bert_model, tokenizer=bert_tokenizer,
                                     pref=bert_prefix)
        chunk = add_doc2vec_embeds(chunk, 'title', fillna='продать', model_path=f'{ROOT_MODELS}/doc2vec_title',
                                   pref=doc2vec_prefix)
        chunk = add_tiny_bert_embeds(chunk, 'description', fillna='продать', model=bert_model, tokenizer=bert_tokenizer,
                                     pref=bert_prefix)
        chunk = add_doc2vec_embeds(chunk, 'description', fillna='продать',
                                   model_path=f'{ROOT_MODELS}/doc2vec_description',
                                   pref=doc2vec_prefix)
        chunk = images_embedding(chunk, 'image', paths=image_paths, w=IMG_HEIGHT, h=IMG_WIDTH, model=image_model,
                                 pref=image_prefix, batch_size=image_batch_size)

        is_valid = (i * batch_size >= (train_size - valid_size)) and name != 'test'
        header = (i == 0) or (is_valid and first_valid)
        mode = 'w' if (i == 0) or (is_valid and first_valid) else 'a'

        if is_valid and first_valid:
            first_valid = False

        if not is_valid:
            chunk.to_csv(f'{ROOT_DATA}/{name}_dataset.csv', index=False, header=header, mode=mode)
        else:
            chunk.to_csv(f'{ROOT_DATA}/valid_dataset.csv', index=False, header=header, mode=mode)


add_embeddings('train')
add_embeddings('test')


def create_tabnet_model():

    first_row = pd.read_csv(f'{ROOT_DATA}/train_dataset.csv', nrows=1)
    n_features = len(get_items_starts_with(first_row.columns, feature_columns))

    train = get_tensorflow_dataset(f'{ROOT_DATA}/train_dataset.csv', batch_size=tabnet_batch_size,
                                   label=label,
                                   feature_columns=feature_columns,
                                   shuffle_buffer_size=tabnet_shuffle_size).prefetch(10)
    valid = get_tensorflow_dataset(f'{ROOT_DATA}/valid_dataset.csv', batch_size=tabnet_batch_size,
                                   label=label,
                                   feature_columns=feature_columns,
                                   shuffle_buffer_size=tabnet_shuffle_size).prefetch(10)
    # Use Group Normalization for small batch sizes
    model = tabnet.TabNetRegressor(feature_columns=None,
                                   num_features=n_features,
                                   num_regressors=1,
                                   num_decision_steps=tabnet_layers,
                                   relaxation_factor=tabnet_feature_relaxation,
                                   sparsity_coefficient=tabnet_sparsity_coeff,
                                   batch_momentum=tabnet_batch_momentum,
                                   virtual_batch_size=tabnet_virtual_batch_size,
                                   feature_dim=tabnet_feature_dim,
                                   output_dim=tabnet_output_dim)

    lr = tf.keras.optimizers.schedules.ExponentialDecay(tabnet_lr, decay_steps=2000, decay_rate=0.95, staircase=False)
    optimizer = tf.keras.optimizers.Adam(lr)
    model.compile(optimizer, loss='mse', metrics=['mse', 'mae'])

    model.fit(train, epochs=tabnet_epochs, validation_data=valid, verbose=True, steps_per_epoch=tabnet_steps_epoch,
              validation_steps=tabnet_valid_steps)

    model.summary()

    return model


tabnet_model = create_tabnet_model()
tabnet_model.save(f'{ROOT_MODELS}/tabnet')

test_chunks = pd.read_csv(f'{ROOT_DATA}/test_dataset.csv', chunksize=batch_size)
for i, test_chunk in enumerate(test_chunks):
    features = get_items_starts_with(test_chunks.columns, feature_columns)
    test_chunk[label] = pd.Series(tabnet_model(tf.convert_to_tensor(test_chunks[features].values)).numpy())
    test_chunks[['item_id', label]].to_csv(f'{ROOT_DATA}/tabnet_submission.csv', index=False, header=i == 0,
                                           mode='w' if i == 0 else 'a')
