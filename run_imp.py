import pandas as pd
import numpy as np
from imputation.generator_discriminator import imputation
from imputation.utils import binary_sampler, min_max_normalization
from imputation.utils_data import load_data_params


def run_imp(data_name, missing_rate):
    data_x, parameters = load_data_params(data_name)
    numerics = np.append(parameters['numeric_cols'], parameters['ordinal_cols'])
    training_cols = np.append(numerics, parameters['categorical_cols'])
    miss_data = data_x.loc[:, training_cols].copy()
    no, dim = miss_data.shape
    hold_out_cols = np.setdiff1d(np.array(data_x.columns), training_cols)

    for miss_rate in missing_rate:
        rmse_list = []
        for it in range(10):
            # Introduce missing data
            data_m = binary_sampler(1 - miss_rate, no, dim, seed=100 * it)
            miss_data_x = miss_data.copy()
            miss_data_x[data_m == 0] = np.nan
            miss_data_x[hold_out_cols] = data_x.loc[:, hold_out_cols].copy()

            training_result = imputation(miss_data_x, parameters=parameters)

            # activate when categorical_list is not empty
            if parameters['categorical_cols']:
                training_result.loc[:, parameters['categorical_cols']] = training_result[parameters['categorical_cols']].apply(pd.to_numeric, errors='coerce')

            imputed_data, _ = min_max_normalization(training_result.loc[:, training_cols])
            ori_data, _ = min_max_normalization(data_x.loc[:, training_cols])

            # Only for missing values
            nominator = np.sum(((1 - data_m) * ori_data - (1 - data_m) * imputed_data) ** 2)
            denominator = np.sum(1 - data_m)

            rmse = np.sqrt(np.sum(nominator) / float(denominator))
            print(str(it + 1) + "-th " + "rmse: " + str(np.round(rmse, 4)))
            rmse_list.append(rmse)

        print("")
        print("Missing Rate: " + str(miss_rate))
        print("normalized_rmse_list: " + str(rmse_list))
        print("normalized_rmse_avg: " + str(sum(rmse_list) / len(rmse_list)))
        print("")


if __name__ == '__main__':
    # ['complete_wine', 'news', 'diabetes', 'letter', 'spam', 'breast_cancer', 'credit_card']
    data_name = "breast_cancer"
    missing_rate = [0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]

    run_imp(data_name, missing_rate)
