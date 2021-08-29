import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from imputation.utils import sample_batch_index, preprocessing, rounding
from imputation.pre_training import pre_training
from imputation.utils_imp import *

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
    set_pre_encoder(theta_pre_Enc, encoder_weights, trainable=False)

    # Encoder weights
    theta_Enc = []
    set_encoder(theta_Enc, encoder_weights, trainable=True)

    # Decoder weights
    theta_Dec = []
    set_dec(theta_Dec, decoder_weights, numeric_dim, categorical_cols, trainable=False)

    theta_Disc = []
    set_disc(theta_Disc, dim)

    generated_sample, dec = generator(X, theta_Enc, theta_Dec, numeric_dim)
    Hat_X = encoder(X, theta_Enc)
    D_prob_imputed = discriminator(Hat_X, theta_Disc)
    D_prob_complete = discriminator(pre_encoder(CX, theta_pre_Enc), theta_Disc)

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



