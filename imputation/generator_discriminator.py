import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from functools import reduce
from tqdm import tqdm

from imputation.utils import xavier_init, sample_batch_index, preprocessing, rounding
from imputation.pre_training import pre_training


def imputation(data, parameters):
    '''
    Impute missing values in data

    data: original data with missing values
    parameters: Generator-Discriminator network parameters:
        - numeric_cols: Numeric columns
        - ordinal_cols: Ordinal columns
        - categorical_cols: Categorical columns
        - pre_batch_size: Mini-batch size for pre-training
        - pre_epochs: Epochs for pre-training
        - pre_learning_rate: Learning rate for pre-training
        - batch_size: Mini-batch size
        - epochs: Epochs
        - learning_rate: Learning rate
        - alpha: alpha for G_loss

    Returns: imputed data
    '''

    numeric_cols = np.append(parameters['numeric_cols'], parameters['ordinal_cols'])
    categorical_cols = parameters['categorical_cols']
    batch_size = parameters['batch_size']
    epochs = parameters['epochs']
    learning_rate = parameters['learning_rate']
    alpha = parameters['alpha']

    ori_data = data
    numerics = np.append(parameters['numeric_cols'], parameters['ordinal_cols'])
    training_cols = np.append(numerics, parameters['categorical_cols'])
    data = data.loc[:, training_cols]
    hold_out_cols = np.setdiff1d(np.array(ori_data.columns), training_cols)

    preprocessed_data = preprocessing(data, {'numeric_cols': numeric_cols,
                                             'categorical_cols': categorical_cols})

    # complete and incomplete data
    complete_data = preprocessed_data['complete_data']
    incomplete_data = preprocessed_data['incomplete_data']

    categorical_lens = preprocessed_data['categorical_lens']

    # pre-training data and parameters
    pre_training_data = pd.concat([complete_data, incomplete_data], axis=0, join='inner').sort_index()
    pre_parameters = {'batch_size': parameters['pre_batch_size'],
                      'epochs': parameters['pre_epochs'],
                      'learning_rate': parameters['pre_learning_rate'],
                      'numeric_cols': numeric_cols,
                      'categorical_cols': categorical_cols,
                      'categorical_lens': categorical_lens}

    # pre-training weights
    pre_weights = pre_training(data=pre_training_data, parameters=pre_parameters)
    encoder_weights = pre_weights['encoder_weights']
    decoder_weights = pre_weights['decoder_weights']

    # Define mask matrix
    mask = 1 - np.isnan(incomplete_data)
    pre_training_mask = 1 - np.isnan(pre_training_data)
    incomplete_data = incomplete_data.fillna(0)

    no = incomplete_data.shape[0]
    numeric_dim = len(numeric_cols)
    categorical_dim = sum(categorical_lens)
    dim = numeric_dim + categorical_dim

    # Input placeholder
    # Incomplete Data vector
    X = tf.placeholder(dtype=tf.float32, shape=[None, dim])
    # Complete Data vector
    CX = tf.placeholder(dtype=tf.float32, shape=[None, dim])
    # Mask vector
    M = tf.placeholder(dtype=tf.float32, shape=[None, dim])

    # Generator variables
    # pre_trained Encoder weights
    theta_pre_Enc = []
    pre_enc_train = False
    theta_pre_Enc.append([tf.Variable(initial_value=encoder_weights[0], name='Encoder_W0', trainable=pre_enc_train),
                          tf.Variable(initial_value=encoder_weights[1], name='Encoder_B0', trainable=pre_enc_train)])
    theta_pre_Enc.append([tf.Variable(initial_value=encoder_weights[2], name='Encoder_W1', trainable=pre_enc_train),
                          tf.Variable(initial_value=encoder_weights[3], name='Encoder_B1', trainable=pre_enc_train)])
    theta_pre_Enc.append([tf.Variable(initial_value=encoder_weights[4], name='Encoder_W2', trainable=pre_enc_train),
                          tf.Variable(initial_value=encoder_weights[5], name='Encoder_B2', trainable=pre_enc_train)])
    theta_pre_Enc.append([tf.Variable(initial_value=encoder_weights[6], name='Encoder_W3', trainable=pre_enc_train),
                          tf.Variable(initial_value=encoder_weights[7], name='Encoder_B3', trainable=pre_enc_train)])
    theta_pre_Enc.append([tf.Variable(initial_value=encoder_weights[8], name='Encoder_W4', trainable=pre_enc_train),
                          tf.Variable(initial_value=encoder_weights[9], name='Encoder_B4', trainable=pre_enc_train)])
    theta_pre_Enc.append([tf.Variable(initial_value=encoder_weights[10], name='Encoder_W5', trainable=pre_enc_train),
                          tf.Variable(initial_value=encoder_weights[11], name='Encoder_B5', trainable=pre_enc_train)])
    # theta_pre_Enc.append([tf.Variable(initial_value=encoder_weights[12], name='Encoder_W6', trainable=pre_enc_train),
    #                       tf.Variable(initial_value=encoder_weights[13], name='Encoder_B6', trainable=pre_enc_train)])
    # theta_pre_Enc.append([tf.Variable(initial_value=encoder_weights[14], name='Encoder_W7', trainable=pre_enc_train),
    #                       tf.Variable(initial_value=encoder_weights[15], name='Encoder_B7', trainable=pre_enc_train)])
    # theta_pre_Enc.append([tf.Variable(initial_value=encoder_weights[16], name='Encoder_W6', trainable=pre_enc_train),
    #                       tf.Variable(initial_value=encoder_weights[17], name='Encoder_B6', trainable=pre_enc_train)])
    # theta_pre_Enc.append([tf.Variable(initial_value=encoder_weights[18], name='Encoder_W7', trainable=pre_enc_train),
    #                       tf.Variable(initial_value=encoder_weights[19], name='Encoder_B7', trainable=pre_enc_train)])

    # Encoder weights
    theta_Enc = []
    enc_train = True
    theta_Enc.append([tf.Variable(initial_value=encoder_weights[0], name='Encoder_W0', trainable=enc_train),
                      tf.Variable(initial_value=encoder_weights[1], name='Encoder_B0', trainable=enc_train)])
    theta_Enc.append([tf.Variable(initial_value=encoder_weights[2], name='Encoder_W1', trainable=enc_train),
                      tf.Variable(initial_value=encoder_weights[3], name='Encoder_B1', trainable=enc_train)])
    theta_Enc.append([tf.Variable(initial_value=encoder_weights[4], name='Encoder_W2', trainable=enc_train),
                      tf.Variable(initial_value=encoder_weights[5], name='Encoder_B2', trainable=enc_train)])
    theta_Enc.append([tf.Variable(initial_value=encoder_weights[6], name='Encoder_W3', trainable=enc_train),
                      tf.Variable(initial_value=encoder_weights[7], name='Encoder_B3', trainable=enc_train)])
    theta_Enc.append([tf.Variable(initial_value=encoder_weights[8], name='Encoder_W4', trainable=enc_train),
                      tf.Variable(initial_value=encoder_weights[9], name='Encoder_B4', trainable=enc_train)])
    theta_Enc.append([tf.Variable(initial_value=encoder_weights[10], name='Encoder_W5', trainable=enc_train),
                      tf.Variable(initial_value=encoder_weights[11], name='Encoder_B5', trainable=enc_train)])
    # theta_Enc.append([tf.Variable(initial_value=encoder_weights[12], name='Encoder_W6', trainable=enc_train),
    #                   tf.Variable(initial_value=encoder_weights[13], name='Encoder_B6', trainable=enc_train)])
    # theta_Enc.append([tf.Variable(initial_value=encoder_weights[14], name='Encoder_W7', trainable=enc_train),
    #                   tf.Variable(initial_value=encoder_weights[15], name='Encoder_B7', trainable=enc_train)])
    # theta_Enc.append([tf.Variable(initial_value=encoder_weights[16], name='Encoder_W8', trainable=enc_train),
    #                   tf.Variable(initial_value=encoder_weights[17], name='Encoder_B8', trainable=enc_train)])
    # theta_Enc.append([tf.Variable(initial_value=encoder_weights[18], name='Encoder_W9', trainable=enc_train),
    #                   tf.Variable(initial_value=encoder_weights[19], name='Encoder_B9', trainable=enc_train)])

    # Decoder weights
    theta_Dec = []
    dec_train = False
    cat_start_idx = 0
    # decoder weights for numeric data (non-trainable)
    if numeric_dim > 0:
        theta_Dec.append([tf.Variable(initial_value=decoder_weights[0], name='Decoder_Num_W0', trainable=dec_train),
                          tf.Variable(initial_value=decoder_weights[1], name='Decoder_Num_B0', trainable=dec_train)])
        theta_Dec.append([tf.Variable(initial_value=decoder_weights[2], name='Decoder_Num_W1', trainable=dec_train),
                          tf.Variable(initial_value=decoder_weights[3], name='Decoder_Num_B1', trainable=dec_train)])
        theta_Dec.append([tf.Variable(initial_value=decoder_weights[4], name='Decoder_Num_W2', trainable=dec_train),
                          tf.Variable(initial_value=decoder_weights[5], name='Decoder_Num_B2', trainable=dec_train)])
        theta_Dec.append([tf.Variable(initial_value=decoder_weights[6], name='Decoder_Num_W3', trainable=dec_train),
                          tf.Variable(initial_value=decoder_weights[7], name='Decoder_Num_B3', trainable=dec_train)])
        theta_Dec.append([tf.Variable(initial_value=decoder_weights[8], name='Decoder_Num_W4', trainable=dec_train),
                          tf.Variable(initial_value=decoder_weights[9], name='Decoder_Num_B4', trainable=dec_train)])
        theta_Dec.append([tf.Variable(initial_value=decoder_weights[10], name='Decoder_Num_W5', trainable=dec_train),
                          tf.Variable(initial_value=decoder_weights[11], name='Decoder_Num_B5', trainable=dec_train)])
        theta_Dec.append([tf.Variable(initial_value=decoder_weights[12], name='Decoder_Num_W6', trainable=dec_train),
                          tf.Variable(initial_value=decoder_weights[13], name='Decoder_Num_B6', trainable=dec_train)])
        theta_Dec.append([tf.Variable(initial_value=decoder_weights[14], name='Decoder_Num_W7', trainable=dec_train),
                          tf.Variable(initial_value=decoder_weights[15], name='Decoder_Num_B7', trainable=dec_train)])

        cat_start_idx = 16
    # decoder weights for categorical data (non-trainable)
    for idx in range(len(categorical_cols)):
        theta_Dec.append([tf.Variable(initial_value=decoder_weights[cat_start_idx + 16 * idx],
                                      name='Decoder_' + categorical_cols[idx] + '_W1', trainable=dec_train),
                          tf.Variable(initial_value=decoder_weights[cat_start_idx + 1 + 16 * idx],
                                      name='Decoder_' + categorical_cols[idx] + '_B1', trainable=dec_train)])
        theta_Dec.append([tf.Variable(initial_value=decoder_weights[cat_start_idx + 2 + 16 * idx],
                                      name='Decoder_' + categorical_cols[idx] + '_W2', trainable=dec_train),
                          tf.Variable(initial_value=decoder_weights[cat_start_idx + 3 + 16 * idx],
                                      name='Decoder_' + categorical_cols[idx] + '_B2', trainable=dec_train)])
        theta_Dec.append([tf.Variable(initial_value=decoder_weights[cat_start_idx + 4 + 16 * idx],
                                      name='Decoder_' + categorical_cols[idx] + '_W3', trainable=dec_train),
                          tf.Variable(initial_value=decoder_weights[cat_start_idx + 5 + 16 * idx],
                                      name='Decoder_' + categorical_cols[idx] + '_B3', trainable=dec_train)])
        theta_Dec.append([tf.Variable(initial_value=decoder_weights[cat_start_idx + 6 + 16 * idx],
                                      name='Decoder_' + categorical_cols[idx] + '_W1', trainable=dec_train),
                          tf.Variable(initial_value=decoder_weights[cat_start_idx + 7 + 16 * idx],
                                      name='Decoder_' + categorical_cols[idx] + '_B1', trainable=dec_train)])
        theta_Dec.append([tf.Variable(initial_value=decoder_weights[cat_start_idx + 8 + 16 * idx],
                                      name='Decoder_' + categorical_cols[idx] + '_W2', trainable=dec_train),
                          tf.Variable(initial_value=decoder_weights[cat_start_idx + 9 + 16 * idx],
                                      name='Decoder_' + categorical_cols[idx] + '_B2', trainable=dec_train)])
        theta_Dec.append([tf.Variable(initial_value=decoder_weights[cat_start_idx + 10 + 16 * idx],
                                      name='Decoder_' + categorical_cols[idx] + '_W3', trainable=dec_train),
                          tf.Variable(initial_value=decoder_weights[cat_start_idx + 11 + 16 * idx],
                                      name='Decoder_' + categorical_cols[idx] + '_B3', trainable=dec_train)])
        theta_Dec.append([tf.Variable(initial_value=decoder_weights[cat_start_idx + 12 + 16 * idx],
                                      name='Decoder_' + categorical_cols[idx] + '_W3', trainable=dec_train),
                          tf.Variable(initial_value=decoder_weights[cat_start_idx + 13 + 16 * idx],
                                      name='Decoder_' + categorical_cols[idx] + '_B3', trainable=dec_train)])
        theta_Dec.append([tf.Variable(initial_value=decoder_weights[cat_start_idx + 14 + 16 * idx],
                                      name='Decoder_' + categorical_cols[idx] + '_W3', trainable=dec_train),
                          tf.Variable(initial_value=decoder_weights[cat_start_idx + 15 + 16 * idx],
                                      name='Decoder_' + categorical_cols[idx] + '_B3', trainable=dec_train)])

    def pre_encoder(data):
        # Auto-encoder (Generator) structure
        tmp_latent0 = tf.nn.tanh(tf.matmul(data, theta_pre_Enc[0][0]) + theta_pre_Enc[0][1])
        tmp_latent1 = tf.nn.tanh(tf.matmul(tmp_latent0, theta_pre_Enc[1][0]) + theta_pre_Enc[1][1])
        tmp_latent2 = tf.nn.tanh(tf.matmul(tmp_latent1, theta_pre_Enc[2][0]) + theta_pre_Enc[2][1])
        tmp_latent3 = tf.nn.tanh(tf.matmul(tmp_latent2, theta_pre_Enc[3][0]) + theta_pre_Enc[3][1])
        tmp_latent4 = tf.nn.tanh(tf.matmul(tmp_latent3, theta_pre_Enc[4][0]) + theta_pre_Enc[4][1])
        # tmp_latent5 = tf.nn.tanh(tf.matmul(tmp_latent4, theta_pre_Enc[5][0]) + theta_pre_Enc[5][1])
        # tmp_latent6 = tf.nn.tanh(tf.matmul(tmp_latent5, theta_pre_Enc[6][0]) + theta_pre_Enc[6][1])
        # tmp_latent7 = tf.nn.tanh(tf.matmul(tmp_latent6, theta_pre_Enc[7][0]) + theta_pre_Enc[7][1])
        # tmp_latent8 = tf.nn.tanh(tf.matmul(tmp_latent7, theta_pre_Enc[8][0]) + theta_pre_Enc[8][1])
        latent = tf.matmul(tmp_latent4, theta_pre_Enc[5][0]) + theta_pre_Enc[5][1]
        return latent

    def encoder(data):
        # Auto-encoder (Generator) structure
        tmp_latent0 = tf.nn.tanh(tf.matmul(data, theta_Enc[0][0]) + theta_Enc[0][1])
        tmp_latent1 = tf.nn.tanh(tf.matmul(tmp_latent0, theta_Enc[1][0]) + theta_Enc[1][1])
        tmp_latent2 = tf.nn.tanh(tf.matmul(tmp_latent1, theta_Enc[2][0]) + theta_Enc[2][1])
        tmp_latent3 = tf.nn.tanh(tf.matmul(tmp_latent2, theta_Enc[3][0]) + theta_Enc[3][1])
        tmp_latent4 = tf.nn.tanh(tf.matmul(tmp_latent3, theta_Enc[4][0]) + theta_Enc[4][1])
        # tmp_latent5 = tf.nn.tanh(tf.matmul(tmp_latent4, theta_Enc[5][0]) + theta_Enc[5][1])
        # tmp_latent6 = tf.nn.tanh(tf.matmul(tmp_latent5, theta_Enc[6][0]) + theta_Enc[6][1])
        # tmp_latent7 = tf.nn.tanh(tf.matmul(tmp_latent6, theta_Enc[7][0]) + theta_Enc[7][1])
        # tmp_latent8 = tf.nn.tanh(tf.matmul(tmp_latent7, theta_Enc[8][0]) + theta_Enc[8][1])
        latent = tf.matmul(tmp_latent4, theta_Enc[5][0]) + theta_Enc[5][1]
        return latent

    def generator(incomplete_data):
        # incomplete_data: incomplete_data
        # Auto-encoder (Generator) structure
        tmp_latent0 = tf.nn.tanh(tf.matmul(incomplete_data, theta_Enc[0][0]) + theta_Enc[0][1])
        tmp_latent1 = tf.nn.tanh(tf.matmul(tmp_latent0, theta_Enc[1][0]) + theta_Enc[1][1])
        tmp_latent2 = tf.nn.tanh(tf.matmul(tmp_latent1, theta_Enc[2][0]) + theta_Enc[2][1])
        tmp_latent3 = tf.nn.tanh(tf.matmul(tmp_latent2, theta_Enc[3][0]) + theta_Enc[3][1])
        tmp_latent4 = tf.nn.tanh(tf.matmul(tmp_latent3, theta_Enc[4][0]) + theta_Enc[4][1])
        # tmp_latent5 = tf.nn.tanh(tf.matmul(tmp_latent4, theta_Enc[5][0]) + theta_Enc[5][1])
        # tmp_latent6 = tf.nn.tanh(tf.matmul(tmp_latent5, theta_Enc[6][0]) + theta_Enc[6][1])
        # tmp_latent7 = tf.nn.tanh(tf.matmul(tmp_latent6, theta_Enc[7][0]) + theta_Enc[7][1])
        # tmp_latent8 = tf.nn.tanh(tf.matmul(tmp_latent7, theta_Enc[8][0]) + theta_Enc[8][1])
        latent = tf.matmul(tmp_latent4, theta_Enc[5][0]) + theta_Enc[5][1]
        # Decoder structure
        dec = []
        cat_start_idx = 0
        if numeric_dim > 0:
            # add layer for numerics
            numeric_latent0 = tf.nn.tanh(tf.matmul(latent, theta_Dec[0][0]) + theta_Dec[0][1])
            numeric_latent1 = tf.nn.tanh(tf.matmul(numeric_latent0, theta_Dec[1][0]) + theta_Dec[1][1])
            numeric_latent2 = tf.nn.tanh(tf.matmul(numeric_latent1, theta_Dec[2][0]) + theta_Dec[2][1])
            numeric_latent3 = tf.nn.tanh(tf.matmul(numeric_latent2, theta_Dec[3][0]) + theta_Dec[3][1])
            numeric_latent4 = tf.nn.tanh(tf.matmul(numeric_latent3, theta_Dec[4][0]) + theta_Dec[4][1])
            numeric_latent5 = tf.nn.tanh(tf.matmul(numeric_latent4, theta_Dec[5][0]) + theta_Dec[5][1])
            numeric_latent6 = tf.nn.tanh(tf.matmul(numeric_latent5, theta_Dec[6][0]) + theta_Dec[6][1])
            dec.append(tf.matmul(numeric_latent6, theta_Dec[7][0]) + theta_Dec[7][1])
            cat_start_idx = 8
        # add layer for categorical
        for i in range(cat_start_idx, len(theta_Dec), 8):
            dec_latent0 = tf.nn.tanh(tf.matmul(latent, theta_Dec[i][0]) + theta_Dec[i][1])
            dec_latent1 = tf.nn.tanh(tf.matmul(dec_latent0, theta_Dec[i + 1][0]) + theta_Dec[i + 1][1])
            dec_latent2 = tf.nn.tanh(tf.matmul(dec_latent1, theta_Dec[i + 2][0]) + theta_Dec[i + 2][1])
            dec_latent3 = tf.nn.tanh(tf.matmul(dec_latent2, theta_Dec[i + 3][0]) + theta_Dec[i + 3][1])
            dec_latent4 = tf.nn.tanh(tf.matmul(dec_latent3, theta_Dec[i + 4][0]) + theta_Dec[i + 4][1])
            dec_latent5 = tf.nn.tanh(tf.matmul(dec_latent4, theta_Dec[i + 5][0]) + theta_Dec[i + 5][1])
            dec_latent = tf.nn.tanh(tf.matmul(dec_latent5, theta_Dec[i + 6][0]) + theta_Dec[i + 6][1])
            if theta_Dec[i + 7][0].shape[1] > 1:
                dec.append(tf.nn.softmax(tf.matmul(dec_latent, theta_Dec[i + 7][0]) + theta_Dec[i + 7][1]))
            else:
                dec.append(tf.nn.sigmoid(tf.matmul(dec_latent, theta_Dec[i + 7][0]) + theta_Dec[i + 7][1]))
        # Output
        outputs = tf.concat(values=dec, axis=1)
        return outputs, dec

    # Discriminator weights
    l_dim = int(dim * 0.7)
    l1_dim = int(l_dim * 0.7)
    l2_dim = int(l_dim * 0.3)

    theta_Disc = []
    theta_Disc.append([tf.Variable(xavier_init([l_dim, l1_dim]), name='Discriminator_W0'),
                       tf.Variable(tf.zeros([l1_dim]), name='Discriminator_b0')])
    theta_Disc.append([tf.Variable(xavier_init([l1_dim, l1_dim]), name='Discriminator_W1'),
                       tf.Variable(tf.zeros([l1_dim]), name='Discriminator_b1')])
    theta_Disc.append([tf.Variable(xavier_init([l1_dim, l2_dim]), name='Discriminator_W2'),
                       tf.Variable(tf.zeros([l2_dim]), name='Discriminator_b2')])
    theta_Disc.append([tf.Variable(xavier_init([l2_dim, l2_dim]), name='Discriminator_W3'),
                       tf.Variable(tf.zeros([l2_dim]), name='Discriminator_b3')])
    theta_Disc.append([tf.Variable(xavier_init([l2_dim, 1]), name='Discriminator_W4'),
                       tf.Variable(tf.zeros([1]), name='Discriminator_b4')])

    def discriminator(data):
        # imputed_data: outputs (of generator function)
        # Discriminator structure
        Disc_h0 = tf.nn.tanh(tf.matmul(data, theta_Disc[0][0]) + theta_Disc[0][1])
        Disc_h1 = tf.nn.tanh(tf.matmul(Disc_h0, theta_Disc[1][0]) + theta_Disc[1][1])
        Disc_h2 = tf.nn.tanh(tf.matmul(Disc_h1, theta_Disc[2][0]) + theta_Disc[2][1])
        Disc_h3 = tf.nn.tanh(tf.matmul(Disc_h2, theta_Disc[3][0]) + theta_Disc[3][1])
        Disc_prob = tf.nn.sigmoid(tf.matmul(Disc_h3, theta_Disc[4][0]) + theta_Disc[4][1])
        return Disc_prob

    generated_sample, dec = generator(incomplete_data=X)
    Hat_X = encoder(data=X)
    D_prob_imputed = discriminator(data=Hat_X)
    D_prob_complete = discriminator(data=pre_encoder(CX))

    # Loss
    D_loss = -tf.reduce_mean(tf.log(D_prob_complete + 1e-12) + tf.log(1 - D_prob_imputed + 1e-12))
    numeric_loss = tf.reduce_mean(tf.square(tf.convert_to_tensor(
        M[:, :numeric_dim] * X[:, :numeric_dim]) - M[:, :numeric_dim] * dec[0])) / tf.reduce_mean(M[:, :numeric_dim])
    categorical_loss = 0
    for i in range(0, len(categorical_cols)):
        if i == 0:
            cat_tensor = tf.convert_to_tensor(X[:, numeric_dim: numeric_dim + categorical_lens[i]])
            categorical_loss += -tf.reduce_mean(
                M[:, numeric_dim: numeric_dim + categorical_lens[i]] * cat_tensor * tf.log(dec[i + 1] + 1e-8))
        else:
            cat_tensor = tf.convert_to_tensor(
                X[:, numeric_dim + sum(categorical_lens[:i]):numeric_dim + sum(categorical_lens[:i + 1])])
            categorical_loss += -tf.reduce_mean(
                M[:, numeric_dim + sum(categorical_lens[:i]):numeric_dim + sum(categorical_lens[:i + 1])]
                * cat_tensor * tf.log(dec[i + 1] + 1e-8))
    G_MSE_loss = numeric_loss + categorical_loss

    G_adv_loss = -tf.reduce_mean(tf.log(D_prob_imputed + 1e-12))
    G_loss = G_adv_loss + alpha * G_MSE_loss

    # Solver
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    D_solver = optimizer.minimize(D_loss, var_list=theta_Disc)
    G_solver = optimizer.minimize(G_loss, var_list=theta_Enc)

    # Session (Initialization)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print("------------- Start Fine-tuning -------------")

    # Iterations (epochs)
    for it in tqdm(range(int(no / batch_size) * epochs)):

        # Sample batch
        batch_idx_missing = sample_batch_index(no, batch_size)
        batch_idx_complete = sample_batch_index(len(complete_data), batch_size)
        batch_data_missing = pre_training_data.iloc[batch_idx_missing, :].fillna(0)
        batch_data_complete = complete_data.iloc[batch_idx_complete, :]
        batch_mask = pre_training_mask.iloc[batch_idx_missing, :]

        Z_num_mb = np.random.normal(loc=0.0, scale=1e-4, size=[batch_size, numeric_dim])
        Z_cat_mb = np.random.uniform(low=0.0, high=1e-2, size=[batch_size, categorical_dim])
        Z_mb = np.hstack((Z_num_mb, Z_cat_mb))

        batch_data_missing = batch_mask * batch_data_missing + (1 - batch_mask) * Z_mb

        _, D_loss_curr = sess.run([D_solver, D_loss],
                                  feed_dict={X: batch_data_missing, M: batch_mask, CX: batch_data_complete})
        _, G_loss_curr, G_adv_loss_curr, G_MSE_loss_curr = sess.run([G_solver, G_loss, G_adv_loss, G_MSE_loss],
                                                                    feed_dict={X: batch_data_missing, M: batch_mask})

        if (it + 1) % (500 * int(no / batch_size)) == 0 or it == int((no / batch_size) * epochs) - 1:
            print("%d-th epoch D_loss: %0.4f" % ((it + 1) / int(no / batch_size), D_loss_curr))
            print("%d-th epoch G_loss: %0.4f" % ((it + 1) / int(no / batch_size), G_loss_curr))
            print("%d-th epoch G_adv_loss: %0.4f" % ((it + 1) / int(no / batch_size), G_adv_loss_curr))
            print("%d-th epoch G_MSE_loss: %0.4f" % ((it + 1) / int(no / batch_size), G_MSE_loss_curr))

    Z_num_b = np.random.normal(loc=0.0, scale=1e-4, size=[no, numeric_dim])
    Z_cat_b = np.random.uniform(low=0.0, high=1e-2, size=[no, categorical_dim])
    Z_b = np.hstack((Z_num_b, Z_cat_b))

    gen_result = sess.run([generated_sample],
                          feed_dict={X: mask * incomplete_data + (1 - mask) * Z_b})[0]
    incomplete_data = mask * incomplete_data + (1 - mask) * gen_result

    # Convert original data to normalized imputed data
    reg_imputed_data = pd.concat([complete_data, incomplete_data], axis=0, join='inner').sort_index()

    # Denormalize numeric data
    imputed_data = pd.DataFrame(
        data=preprocessed_data['denormalization'].inverse_transform(reg_imputed_data[numeric_cols]),
        index=reg_imputed_data.index, columns=numeric_cols)

    # One-hot decode categorical data
    categorical_cols_tranf = preprocessed_data['categorical_cols_tranf']
    idx = 0
    for classes in categorical_cols_tranf:
        if len(classes) == 1:
            imputed_data[classes] = reg_imputed_data[classes]
            idx += 1
        else:
            category_name = categorical_cols[idx]
            imputed_data[category_name] = reg_imputed_data[classes].idxmax(axis=1).str[len(category_name + "_"):]
            idx += 1

    ref_imp_data = rounding(imputed_data, parameters['ordinal_cols'])
    ref_imp_data[hold_out_cols] = ori_data.loc[:, hold_out_cols]
    return ref_imp_data



