import pandas as pd
import numpy as np
from imputation.generator_discriminator import imputation
from imputation.utils import binary_sampler, min_max_normalization

default_path = "./data/"
data_name = "complete_wine"
data_x = pd.read_csv(default_path + data_name + ".csv", keep_default_na=False)

# complete_wine.csv parameters
categorical_list = []
ordinal_list = ['fixed_acidity', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'pH', 'quality']
numeric_list = list(set(data_x.columns) - set(ordinal_list))
parameters = {
    'numeric_cols': numeric_list,
    'ordinal_cols': ordinal_list,
    'categorical_cols': categorical_list,
    'pre_batch_size': 256,
    'pre_epochs': 1000,
    'pre_learning_rate': 1e-3,
    'batch_size': 256,
    'epochs': 500,
    'learning_rate': 1e-3,
    'alpha': 1}

# news.csv parameters
# ordinal_list = ['n_tokens_title', 'n_tokens_content', 'num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos',
#                 'num_keywords', 'kw_min_min', 'kw_max_min', 'kw_min_max', 'kw_max_max', 'self_reference_min_shares',
#                 'self_reference_max_shares']
# categorical_list = ['data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus',
#                     'data_channel_is_socmed', 'data_channel_is_tech', 'data_channel_is_world',
#                     'weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 'weekday_is_thursday',
#                     'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday', 'is_weekend']
# numeric_list = list(set(data_x.columns) - set(ordinal_list) - set(categorical_list))
# parameters = {
#     'numeric_cols': numeric_list,
#     'ordinal_cols': ordinal_list,
#     'categorical_cols': categorical_list,
#     'pre_batch_size': 256,
#     'pre_epochs': 50,
#     'pre_learning_rate': 1e-3,
#     'batch_size': 256,
#     'epochs': 20,
#     'learning_rate': 1e-4,
#     'alpha': 1}

# diabetes.csv parameter
# categorical_list = []
# numeric_list = ['BMI', 'DiabetesPedigreeFunction']
# ordinal_list = list(set(data_x.columns) - set(numeric_list))
# parameters = {
#     'numeric_cols': numeric_list,
#     'ordinal_cols': ordinal_list,
#     'categorical_cols': categorical_list,
#     'pre_batch_size': 256,
#     'pre_epochs': 1000,
#     'pre_learning_rate': 1e-3,
#     'batch_size': 128,
#     'epochs': 500,
#     'learning_rate': 1e-4,
#     'alpha': 10}

# letter.csv parameters
# ordinal_list = list(data_x.columns)
# categorical_list = []
# numeric_list = []
# parameters = {
#     'numeric_cols': numeric_list,
#     'ordinal_cols': ordinal_list,
#     'categorical_cols': categorical_list,
#     'pre_batch_size': 256,
#     'pre_epochs': 1000,
#     'pre_learning_rate': 1e-3,
#     'batch_size': 256,
#     'epochs': 500,
#     'learning_rate': 1e-3,
#     'alpha': 10}

# spam.csv parameters
# ordinal_list = []
# categorical_list = []
# numeric_list = list(data_x.columns)
# parameters = {
#     'numeric_cols': numeric_list,
#     'ordinal_cols': ordinal_list,
#     'categorical_cols': categorical_list,
#     'pre_batch_size': 256,
#     'pre_epochs': 20,
#     'pre_learning_rate': 1e-4,
#     'batch_size': 256,
#     'epochs': 100,
#     'learning_rate': 1e-4,
#     'alpha': 10}

# breast_cancer.csv parameters
# ordinal_list = []
# categorical_list = []
# numeric_list = list(data_x.columns)
# parameters = {
#     'numeric_cols': numeric_list,
#     'ordinal_cols': ordinal_list,
#     'categorical_cols': categorical_list,
#     'pre_batch_size': 256,
#     'pre_epochs': 1000,
#     'pre_learning_rate': 1e-3,
#     'batch_size': 56,
#     'epochs': 500,
#     'learning_rate': 1e-4,
#     'alpha': 10}

# credit_card.csv parameters
# numeric_list = []
# categorical_list = ['SEX']
# ordinal_list = list(set(data_x.columns) - set(categorical_list))
# parameters = {
#     'numeric_cols': numeric_list,
#     'ordinal_cols': ordinal_list,
#     'categorical_cols': categorical_list,
#     'pre_batch_size': 256,
#     'pre_epochs': 200,
#     'pre_learning_rate': 1e-3,
#     'batch_size': 256,
#     'epochs': 50,
#     'learning_rate': 1e-4,
#     'alpha': 10}

numerics = np.append(parameters['numeric_cols'], parameters['ordinal_cols'])
training_cols = np.append(numerics, parameters['categorical_cols'])
miss_data = data_x.loc[:, training_cols].copy()
no, dim = miss_data.shape
hold_out_cols = np.setdiff1d(np.array(data_x.columns), training_cols)

miss_rate_list = [0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
for miss_rate in miss_rate_list:
    rmse_list = []
    for it in range(10):
        # Introduce missing data
        data_m = binary_sampler(1 - miss_rate, no, dim, seed=100 * it)
        miss_data_x = miss_data.copy()
        miss_data_x[data_m == 0] = np.nan
        miss_data_x[hold_out_cols] = data_x.loc[:, hold_out_cols].copy()

        training_result = imputation(miss_data_x, parameters=parameters)

        # activate when categorical_list is not empty
        if categorical_list:
            training_result.loc[:, parameters['categorical_cols']] = \
                training_result[parameters['categorical_cols']].apply(pd.to_numeric, errors='coerce')

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
    print("rmse_list: " + str(rmse_list))
    print("rmse_avg: " + str(sum(rmse_list) / len(rmse_list)))
    print("")

