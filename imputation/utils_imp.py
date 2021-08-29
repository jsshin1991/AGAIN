import tensorflow as tf
from imputation.utils import xavier_init


def set_pre_encoder(theta_pre_Enc, init_weights, trainable=False):
    for idx in range(0, 12, 2):
        theta_pre_Enc.append([tf.Variable(initial_value=init_weights[idx], name='Encoder_W' + str(idx//2), trainable=trainable),
                              tf.Variable(initial_value=init_weights[idx + 1], name='Encoder_B' + str(idx//2), trainable=trainable)])
    return None

def set_encoder(theta_Enc, init_weights, trainable=True):
    for idx in range(0, 12, 2):
        theta_Enc.append([tf.Variable(initial_value=init_weights[idx], name='Encoder_W' + str(idx//2), trainable=trainable),
                          tf.Variable(initial_value=init_weights[idx + 1], name='Encoder_B' + str(idx//2), trainable=trainable)])
    return None


def set_dec(theta_Dec, init_weights, numeric_dim, categorical_cols, trainable):
    cat_start_idx = 0
    # decoder weights for numeric data (non-trainable)
    if numeric_dim > 0:
        for idx in range(0, 16, 2):
            theta_Dec.append([tf.Variable(initial_value=init_weights[idx], name='Decoder_Num_W' + str(idx//2), trainable=trainable),
                              tf.Variable(initial_value=init_weights[idx+1], name='Decoder_Num_B' + str(idx//2), trainable=trainable)])
        cat_start_idx = 16
    # decoder weights for categorical data (non-trainable)
    for idx in range(len(categorical_cols)):
        for in_idx in range(0, 16, 2):
            theta_Dec.append([tf.Variable(initial_value=init_weights[cat_start_idx + in_idx + 16 * idx],
                                          name='Decoder_' + categorical_cols[idx] + '_W' + str(in_idx//2), trainable=trainable),
                              tf.Variable(initial_value=init_weights[cat_start_idx + (in_idx + 1) + 16 * idx],
                                          name='Decoder_' + categorical_cols[idx] + '_B' + str(in_idx//2), trainable=trainable)])
    return None

def set_disc(theta_Disc, dim):
    # Discriminator weights
    l_dim = int(dim * 0.7)
    l1_dim = int(l_dim * 0.7)
    l2_dim = int(l_dim * 0.3)

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
    return None

def pre_encoder(data, theta_pre_Enc):
    # Auto-encoder (Generator) structure
    tmp_latent0 = tf.nn.tanh(tf.matmul(data, theta_pre_Enc[0][0]) + theta_pre_Enc[0][1])
    tmp_latent1 = tf.nn.tanh(tf.matmul(tmp_latent0, theta_pre_Enc[1][0]) + theta_pre_Enc[1][1])
    tmp_latent2 = tf.nn.tanh(tf.matmul(tmp_latent1, theta_pre_Enc[2][0]) + theta_pre_Enc[2][1])
    tmp_latent3 = tf.nn.tanh(tf.matmul(tmp_latent2, theta_pre_Enc[3][0]) + theta_pre_Enc[3][1])
    tmp_latent4 = tf.nn.tanh(tf.matmul(tmp_latent3, theta_pre_Enc[4][0]) + theta_pre_Enc[4][1])
    latent = tf.matmul(tmp_latent4, theta_pre_Enc[5][0]) + theta_pre_Enc[5][1]
    return latent

def encoder(data, theta_Enc):
    # Auto-encoder (Generator) structure
    tmp_latent0 = tf.nn.tanh(tf.matmul(data, theta_Enc[0][0]) + theta_Enc[0][1])
    tmp_latent1 = tf.nn.tanh(tf.matmul(tmp_latent0, theta_Enc[1][0]) + theta_Enc[1][1])
    tmp_latent2 = tf.nn.tanh(tf.matmul(tmp_latent1, theta_Enc[2][0]) + theta_Enc[2][1])
    tmp_latent3 = tf.nn.tanh(tf.matmul(tmp_latent2, theta_Enc[3][0]) + theta_Enc[3][1])
    tmp_latent4 = tf.nn.tanh(tf.matmul(tmp_latent3, theta_Enc[4][0]) + theta_Enc[4][1])
    latent = tf.matmul(tmp_latent4, theta_Enc[5][0]) + theta_Enc[5][1]
    return latent

def generator(data, theta_Enc, theta_Dec, numeric_dim):
    # Auto-encoder (Generator) structure
    tmp_latent0 = tf.nn.tanh(tf.matmul(data, theta_Enc[0][0]) + theta_Enc[0][1])
    tmp_latent1 = tf.nn.tanh(tf.matmul(tmp_latent0, theta_Enc[1][0]) + theta_Enc[1][1])
    tmp_latent2 = tf.nn.tanh(tf.matmul(tmp_latent1, theta_Enc[2][0]) + theta_Enc[2][1])
    tmp_latent3 = tf.nn.tanh(tf.matmul(tmp_latent2, theta_Enc[3][0]) + theta_Enc[3][1])
    tmp_latent4 = tf.nn.tanh(tf.matmul(tmp_latent3, theta_Enc[4][0]) + theta_Enc[4][1])
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


def discriminator(data, theta_Disc):
    # imputed_data: outputs (of generator function)
    # Discriminator structure
    Disc_h0 = tf.nn.tanh(tf.matmul(data, theta_Disc[0][0]) + theta_Disc[0][1])
    Disc_h1 = tf.nn.tanh(tf.matmul(Disc_h0, theta_Disc[1][0]) + theta_Disc[1][1])
    Disc_h2 = tf.nn.tanh(tf.matmul(Disc_h1, theta_Disc[2][0]) + theta_Disc[2][1])
    Disc_h3 = tf.nn.tanh(tf.matmul(Disc_h2, theta_Disc[3][0]) + theta_Disc[3][1])
    Disc_prob = tf.nn.sigmoid(tf.matmul(Disc_h3, theta_Disc[4][0]) + theta_Disc[4][1])
    return Disc_prob