import tensorflow as tf
import numpy as np
import pandas as pd
from functools import reduce
from tqdm import tqdm

from imputation.utils import xavier_init, sample_batch_index


def pre_training(data, parameters):
    '''
    Auto-encoder for pre-training

    data: original data with missing values
    parameters: Auto-encoder network parameters:
        - numeric_cols: Numeric columns
        - categorical_cols: Categorical columns
        - batch_size: Mini-batch size
        - epochs: Epochs
        - learning_rate: Learning rate
        - categorical_lens: Categorical Lens

    Returns: {'encoder_weights': encoder_weights_tensor, 'decoder_weights': decoder_weights_tensor}
    '''

    numeric_cols = parameters['numeric_cols']
    categorical_cols = parameters['categorical_cols']
    batch_size = parameters['batch_size']
    epochs = parameters['epochs']
    learning_rate = parameters['learning_rate']
    categorical_lens = parameters['categorical_lens']

    mask = 1 - np.isnan(data)
    data_x = data.fillna(0)

    no = data_x.shape[0]
    numeric_dim = len(numeric_cols)
    categorical_dim = sum(categorical_lens)
    dim = numeric_dim + categorical_dim

    # Dimension of the encoder layers
    h0_dim = int(dim * 0.9)
    h1_dim = int(dim * 0.8)
    l_dim = int(dim * 0.7)

    # Input placeholder
    # Data vector
    X = tf.placeholder(dtype=tf.float32, shape=[None, dim])
    # Mask vector
    M = tf.placeholder(dtype=tf.float32, shape=[None, dim])

    # Encoder weights
    theta_Enc = []
    theta_Enc.append([tf.Variable(initial_value=xavier_init([dim, h0_dim]), name='Encoder_W0'),
                      tf.Variable(tf.zeros(shape=[h0_dim]), name='Encoder_B0')])
    theta_Enc.append([tf.Variable(initial_value=xavier_init([h0_dim, h0_dim]), name='Encoder_W1'),
                      tf.Variable(tf.zeros(shape=[h0_dim]), name='Encoder_B1')])
    theta_Enc.append([tf.Variable(initial_value=xavier_init([h0_dim, h1_dim]), name='Encoder_W2'),
                      tf.Variable(tf.zeros(shape=[h1_dim]), name='Encoder_B2')])
    theta_Enc.append([tf.Variable(initial_value=xavier_init([h1_dim, h1_dim]), name='Encoder_W3'),
                      tf.Variable(tf.zeros(shape=[h1_dim]), name='Encoder_B3')])
    # theta_Enc.append([tf.Variable(initial_value=xavier_init([h1_dim, h2_dim]), name='Encoder_W4'),
    #                   tf.Variable(tf.zeros(shape=[h2_dim]), name='Encoder_B4')])
    # theta_Enc.append([tf.Variable(initial_value=xavier_init([h2_dim, h2_dim]), name='Encoder_W5'),
    #                   tf.Variable(tf.zeros(shape=[h2_dim]), name='Encoder_B5')])
    # theta_Enc.append([tf.Variable(initial_value=xavier_init([h2_dim, h3_dim]), name='Encoder_W6'),
    #                   tf.Variable(tf.zeros(shape=[h3_dim]), name='Encoder_B6')])
    # theta_Enc.append([tf.Variable(initial_value=xavier_init([h3_dim, h3_dim]), name='Encoder_W7'),
    #                   tf.Variable(tf.zeros(shape=[h3_dim]), name='Encoder_B7')])
    theta_Enc.append([tf.Variable(initial_value=xavier_init([h1_dim, l_dim]), name='Encoder_W8'),
                      tf.Variable(tf.zeros(shape=[l_dim]), name='Encoder_B8')])
    theta_Enc.append([tf.Variable(initial_value=xavier_init([l_dim, l_dim]), name='Encoder_W9'),
                      tf.Variable(tf.zeros(shape=[l_dim]), name='Encoder_B9')])
    # theta_Enc.append([tf.Variable(initial_value=xavier_init([h0_dim, h0_dim]), name='Encoder_W1'),
    #                   tf.Variable(tf.zeros(shape=[h0_dim]), name='Encoder_B1')])
    # theta_Enc.append([tf.Variable(initial_value=xavier_init([h0_dim, l_dim]), name='Encoder_W2'),
    #                   tf.Variable(tf.zeros(shape=[l_dim]), name='Encoder_B2')])
    # theta_Enc.append([tf.Variable(initial_value=xavier_init([l_dim, l_dim]), name='Encoder_W3'),
    #                   tf.Variable(tf.zeros(shape=[l_dim]), name='Encoder_B3')])
    # theta_Enc.append([tf.Variable(initial_value=xavier_init([h1_dim, l_dim]), name='Encoder_W4'),
    #                   tf.Variable(tf.zeros(shape=[l_dim]), name='Encoder_B4')])
    # theta_Enc.append([tf.Variable(initial_value=xavier_init([l_dim, l_dim]), name='Encoder_W5'),
    #                   tf.Variable(tf.zeros(shape=[l_dim]), name='Encoder_B5')])
    # theta_Enc.append([tf.Variable(initial_value=xavier_init([h2_dim, l_dim]), name='Encoder_W6'),
    #                   tf.Variable(tf.zeros(shape=[l_dim]), name='Encoder_B6')])
    # theta_Enc.append([tf.Variable(initial_value=xavier_init([l_dim, l_dim]), name='Encoder_W7'),
    #                   tf.Variable(tf.zeros(shape=[l_dim]), name='Encoder_B7')])

    # Decoder weights
    theta_Dec = []
    # decoder weights for numeric data
    if numeric_dim > 0:
        base_dim = l_dim + numeric_dim
        base_dim_7 = int(0.7 * base_dim)
        base_dim_3 = int(0.3 * base_dim)
        base_dim_5 = int(0.5 * base_dim)
        # theta_Dec.append([tf.Variable(initial_value=xavier_init([l_dim, l_dim]), name='Decoder_Num_W0'),
        #                   tf.Variable(tf.zeros(shape=[l_dim]), name='Decoder_Num_B0')])
        # theta_Dec.append([tf.Variable(initial_value=xavier_init([l_dim, base_dim_5]), name='Decoder_Num_W3'),
        #                   tf.Variable(tf.zeros(shape=[base_dim_5]), name='Decoder_Num_B3')])
        # theta_Dec.append([tf.Variable(initial_value=xavier_init([base_dim_5, base_dim_5]), name='Decoder_Num_W4'),
        #                   tf.Variable(tf.zeros(shape=[base_dim_5]), name='Decoder_Num_B4')])
        # theta_Dec.append([tf.Variable(initial_value=xavier_init([base_dim_5, numeric_dim]), name='Decoder_Num_W5'),
        #                   tf.Variable(tf.zeros(shape=[numeric_dim]), name='Decoder_Num_B5')])

        if l_dim > numeric_dim:
            theta_Dec.append([tf.Variable(initial_value=xavier_init([l_dim, base_dim_7]), name='Decoder_Num_W3'),
                              tf.Variable(tf.zeros(shape=[base_dim_7]), name='Decoder_Num_B3')])
            theta_Dec.append([tf.Variable(initial_value=xavier_init([base_dim_7, base_dim_7]), name='Decoder_Num_W4'),
                              tf.Variable(tf.zeros(shape=[base_dim_7]), name='Decoder_Num_B4')])
            theta_Dec.append([tf.Variable(initial_value=xavier_init([base_dim_7, base_dim_5]), name='Decoder_Num_W5'),
                              tf.Variable(tf.zeros(shape=[base_dim_5]), name='Decoder_Num_B5')])
            theta_Dec.append([tf.Variable(initial_value=xavier_init([base_dim_5, base_dim_5]), name='Decoder_Num_W6'),
                              tf.Variable(tf.zeros(shape=[base_dim_5]), name='Decoder_Num_B6')])
            theta_Dec.append([tf.Variable(initial_value=xavier_init([base_dim_5, base_dim_3]), name='Decoder_Num_W7'),
                              tf.Variable(tf.zeros(shape=[base_dim_3]), name='Decoder_Num_B7')])
            theta_Dec.append([tf.Variable(initial_value=xavier_init([base_dim_3, base_dim_3]), name='Decoder_Num_W8'),
                              tf.Variable(tf.zeros(shape=[base_dim_3]), name='Decoder_Num_B8')])
            theta_Dec.append([tf.Variable(initial_value=xavier_init([base_dim_3, numeric_dim]), name='Decoder_Num_W9'),
                              tf.Variable(tf.zeros(shape=[numeric_dim]), name='Decoder_Num_B9')])
        else:
            theta_Dec.append([tf.Variable(initial_value=xavier_init([l_dim, base_dim_3]), name='Decoder_Num_W3'),
                              tf.Variable(tf.zeros(shape=[base_dim_3]), name='Decoder_Num_B3')])
            theta_Dec.append([tf.Variable(initial_value=xavier_init([base_dim_3, base_dim_3]), name='Decoder_Num_W4'),
                              tf.Variable(tf.zeros(shape=[base_dim_3]), name='Decoder_Num_B4')])
            theta_Dec.append([tf.Variable(initial_value=xavier_init([base_dim_3, base_dim_5]), name='Decoder_Num_W5'),
                              tf.Variable(tf.zeros(shape=[base_dim_5]), name='Decoder_Num_B5')])
            theta_Dec.append([tf.Variable(initial_value=xavier_init([base_dim_5, base_dim_5]), name='Decoder_Num_W6'),
                              tf.Variable(tf.zeros(shape=[base_dim_5]), name='Decoder_Num_B6')])
            theta_Dec.append([tf.Variable(initial_value=xavier_init([base_dim_5, base_dim_7]), name='Decoder_Num_W7'),
                              tf.Variable(tf.zeros(shape=[base_dim_7]), name='Decoder_Num_B7')])
            theta_Dec.append([tf.Variable(initial_value=xavier_init([base_dim_7, base_dim_7]), name='Decoder_Num_W8'),
                              tf.Variable(tf.zeros(shape=[base_dim_7]), name='Decoder_Num_B8')])
            theta_Dec.append([tf.Variable(initial_value=xavier_init([base_dim_7, numeric_dim]), name='Decoder_Num_W9'),
                              tf.Variable(tf.zeros(shape=[numeric_dim]), name='Decoder_Num_B9')])
        # theta_Dec.append([tf.Variable(initial_value=xavier_init([l_dim, numeric_dim]), name='Decoder_Num_W1'),
        #                   tf.Variable(tf.zeros(shape=[numeric_dim]), name='Decoder_Num_B1')])
        theta_Dec.append([tf.Variable(initial_value=xavier_init([numeric_dim, numeric_dim]), name='Decoder_Num_W2'),
                          tf.Variable(tf.zeros(shape=[numeric_dim]), name='Decoder_Num_B2')])

    # decoder weights for categorical data
    for idx in range(len(categorical_cols)):
        cat_c = categorical_cols[idx]
        cat_dim = categorical_lens[idx]
        base_c_dim = l_dim + cat_dim
        c_dim_7 = int(0.7 * base_c_dim)
        c_dim_3 = int(0.3 * base_c_dim)
        c_dim_5 = int(0.5 * base_c_dim)
        # theta_Dec.append([tf.Variable(initial_value=xavier_init([l_dim, l_dim]), name='Decoder_' + cat_c + '_W1'),
        #                   tf.Variable(tf.zeros(shape=[l_dim]), name='Decoder_' + cat_c + '_B1')])
        theta_Dec.append([tf.Variable(initial_value=xavier_init([l_dim, c_dim_7]), name='Decoder_' + cat_c + '_W2'),
                          tf.Variable(tf.zeros(shape=[c_dim_7]), name='Decoder_' + cat_c + '_B2')])
        theta_Dec.append([tf.Variable(initial_value=xavier_init([c_dim_7, c_dim_7]), name='Decoder_' + cat_c + '_W3'),
                          tf.Variable(tf.zeros(shape=[c_dim_7]), name='Decoder_' + cat_c + '_B3')])
        theta_Dec.append([tf.Variable(initial_value=xavier_init([c_dim_7, c_dim_5]), name='Decoder_' + cat_c + '_W3'),
                          tf.Variable(tf.zeros(shape=[c_dim_5]), name='Decoder_' + cat_c + '_B3')])
        theta_Dec.append([tf.Variable(initial_value=xavier_init([c_dim_5, c_dim_5]), name='Decoder_' + cat_c + '_W3'),
                          tf.Variable(tf.zeros(shape=[c_dim_5]), name='Decoder_' + cat_c + '_B3')])
        theta_Dec.append([tf.Variable(initial_value=xavier_init([c_dim_5, c_dim_3]), name='Decoder_' + cat_c + '_W3'),
                          tf.Variable(tf.zeros(shape=[c_dim_3]), name='Decoder_' + cat_c + '_B3')])
        theta_Dec.append([tf.Variable(initial_value=xavier_init([c_dim_3, c_dim_3]), name='Decoder_' + cat_c + '_W3'),
                          tf.Variable(tf.zeros(shape=[c_dim_3]), name='Decoder_' + cat_c + '_B3')])
        theta_Dec.append([tf.Variable(initial_value=xavier_init([c_dim_3, cat_dim]), name='Decoder_' + cat_c + '_W4'),
                          tf.Variable(tf.zeros(shape=[cat_dim]), name='Decoder_' + cat_c + '_B4')])
        theta_Dec.append([tf.Variable(initial_value=xavier_init([cat_dim, cat_dim]), name='Decoder_' + cat_c + '_W4'),
                          tf.Variable(tf.zeros(shape=[cat_dim]), name='Decoder_' + cat_c + '_B4')])

    # Auto-encoder structure
    # Encoder structure
    tmp_latent0 = tf.nn.tanh(tf.matmul(X, theta_Enc[0][0]) + theta_Enc[0][1])
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

    # Loss
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
    loss = numeric_loss + categorical_loss

    # Solver
    theta = reduce(lambda x, y: x + y, theta_Enc + theta_Dec)

    solver = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss=loss, var_list=theta)

    # Session (Initialization)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print("------------- Start Pre-training -------------")

    # Iterations (epochs)
    for it in tqdm(range(int(no / batch_size) * epochs)):
        # Sample batch
        batch_idx = sample_batch_index(no, batch_size)
        batch_data = data_x.iloc[batch_idx, :]
        batch_mask = mask.iloc[batch_idx, :]

        _, loss_curr = sess.run([solver, loss], feed_dict={X: batch_data, M: batch_mask})

        if (it + 1) % (2000 * int(no / batch_size)) == 0 or it == int(no / batch_size) * epochs - 1:
            print("%d-th epoch loss: %0.4f" % ((it + 1) / int(no / batch_size), loss_curr))

    encoder_weights = sess.run(theta[:12])
    decoder_weights = sess.run(theta[12:])

    return {'encoder_weights': encoder_weights, 'decoder_weights': decoder_weights}


