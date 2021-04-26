import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

'''
normalization: MinMax Normalizer
renormalization: Recover the data from normalzied data
xavier_init: Xavier initialization
sample_batch_index: sample random batch index
'''


def min_max_normalization(data):
    '''Normalize data in [0, 1] range.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
      - norm_parameters: min_val, max_val for each feature for renormalization
    '''

    # Parameters
    _, dim = data.shape
    norm_data = data.copy()

    # MixMax normalization
    min_val = np.zeros(dim)
    max_val = np.zeros(dim)

    # For each dimension
    for i in range(dim):
        min_val[i] = np.nanmin(norm_data.iloc[:, i])
        norm_data.iloc[:, i] = norm_data.iloc[:, i] - np.nanmin(norm_data.iloc[:, i])
        max_val[i] = np.nanmax(norm_data.iloc[:, i])
        norm_data.iloc[:, i] = norm_data.iloc[:, i] / (np.nanmax(norm_data.iloc[:, i]) + 1e-6)

    # Return norm_parameters for renormalization
    norm_parameters = {'min_val': min_val,
                       'max_val': max_val}

    return norm_data, norm_parameters


def xavier_init(size):
    '''Xavier initialization.

    Args:
      - size: vector size

    Returns:
      - initialized random vector.
    '''
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def sample_batch_index(total, batch_size):
    '''Sample index of the mini-batch.

    Args:
      - total: total number of samples
      - batch_size: batch size

    Returns:
      - batch_idx: batch index
    '''
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx


def preprocessing(prep_data, prep_parameters):
    '''
    Normalization (for numerics) and one-hot encoding (for categorical variables)

    prep_data: original data with missing values
    prep_parameters:
        - numeric_cols: Numeric columns
        - categorical_cols: Categorical columns

    Returns: {'complete_data': complete preprocessed data,
              'incomplete_data': incomplete preprocessed data,
              'denormalization': for denormalization,
              'categorical_cols_tranf': for one-hot decoding}
    '''

    prep_data = prep_data[prep_data != '']
    numeric_cols = prep_parameters['numeric_cols']
    categorical_cols = prep_parameters['categorical_cols']

    # normalization for numeric variables
    numeric_data = prep_data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    numeric_fit = StandardScaler().fit(numeric_data)
    numeric_transf = numeric_fit.transform(numeric_data)
    numeric_data = pd.DataFrame(numeric_transf, index=prep_data.index, columns=numeric_cols).astype(
        dtype=np.float32)

    categorical_lens = []
    categorical_cols_transf = []
    categorical_data = pd.DataFrame()
    for classes in categorical_cols:
        # one-hot encoding for categorical variables
        class_df = pd.get_dummies(prep_data.loc[:, classes], prefix=classes)
        categorical_cols_transf.append(class_df.columns)
        categorical_lens.append(len(class_df.columns))
        if len(class_df.columns) > 1:
            class_df.loc[class_df.sum(axis=1) == 0, :] = np.nan
        categorical_data = pd.concat([categorical_data, class_df], axis=1).astype(dtype=np.float32)

    prep_data = pd.concat([numeric_data, categorical_data], axis=1)
    complete_prep_data = prep_data.dropna()
    # incomplete_prep_data = prep_data[pd.isnull(prep_data).any(axis=1)]
    incomplete_prep_data = prep_data.iloc[list(set(prep_data.index) - set(complete_prep_data.index)), :]

    return {'complete_data': complete_prep_data,
            'incomplete_data': incomplete_prep_data,
            'categorical_lens': categorical_lens,
            'denormalization': numeric_fit,
            'categorical_cols_tranf': categorical_cols_transf}


def binary_sampler(p, rows, cols, seed=None):
    '''Sample binary random variables.

    Args:
      - p: probability of 1
      - rows: the number of rows
      - cols: the number of columns

    Returns:
      - binary_random_matrix: generated binary random matrix.
    '''

    max_rate_incomp_rows = 0.9
    np.random.seed(seed=seed)
    np.random.seed(seed=np.random.random_integers(low=0, high=1e4))
    missing_data_idx = sample_batch_index(rows, int(rows * max_rate_incomp_rows))
    unif_random_matrix = np.zeros(shape=[rows, cols])
    unif_random_matrix[missing_data_idx, :] = np.random.uniform(0., 1., size=[len(missing_data_idx), cols])

    binary_random_matrix = 1 * (unif_random_matrix < (p - (1 - max_rate_incomp_rows)) / max_rate_incomp_rows)

    return binary_random_matrix


def rounding(imputed_data, ordinal_cols):
    '''Round imputed data for categorical variables.

    Args:
      - imputed_data: imputed data
      - data_x: original data with missing values

    Returns:
      - rounded_data: rounded imputed data
    '''

    rounded_data = imputed_data.copy()

    for col in ordinal_cols:
        rounded_data.loc[:, col] = np.round(rounded_data.loc[:, col])

    return rounded_data

