import pandas as pd


def load_data_params(data_name):
    avail_list = ['complete_wine', 'news', 'diabetes', 'letter', 'spam', 'breast_cancer', 'credit_card']
    check_avail_data(data_name, avail_list)

    default_path = "./data/"
    data_x = pd.read_csv(default_path + data_name + ".csv", keep_default_na=False)
    parameters = {}
    if data_name == 'complete_wine':
        parameters = load_complete_wine_settings(data_x)
    elif data_name == 'news':
        parameters = load_news_settings(data_x)
    elif data_name == 'diabetes':
        parameters = load_diabetes_settings(data_x)
    elif data_name == 'letter':
        parameters = load_letter_settings(data_x)
    elif data_name == 'spam':
        parameters = load_spam_settings(data_x)
    elif data_name == 'breast_cancer':
        parameters = load_breast_cancer_settings(data_x)
    elif data_name == 'credit_card':
        parameters = load_credit_card_settings(data_x)

    return data_x, parameters


def check_avail_data(data_name, avail_list):
    try:
        if data_name not in avail_list:
            raise Exception('There is no corresponding data!!')
    except Exception as e:
        print(e)
    return None

def load_complete_wine_settings(data_x):
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

    return parameters

def load_news_settings(data_x):
    # news.csv parameters
    ordinal_list = ['n_tokens_title', 'n_tokens_content', 'num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos',
                    'num_keywords', 'kw_min_min', 'kw_max_min', 'kw_min_max', 'kw_max_max', 'self_reference_min_shares',
                    'self_reference_max_shares']
    categorical_list = ['data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus',
                        'data_channel_is_socmed', 'data_channel_is_tech', 'data_channel_is_world',
                        'weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 'weekday_is_thursday',
                        'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday', 'is_weekend']
    numeric_list = list(set(data_x.columns) - set(ordinal_list) - set(categorical_list))
    parameters = {
        'numeric_cols': numeric_list,
        'ordinal_cols': ordinal_list,
        'categorical_cols': categorical_list,
        'pre_batch_size': 256,
        'pre_epochs': 50,
        'pre_learning_rate': 1e-3,
        'batch_size': 256,
        'epochs': 20,
        'learning_rate': 1e-4,
        'alpha': 1}

    return parameters

def load_diabetes_settings(data_x):
    # diabetes.csv parameter
    categorical_list = []
    numeric_list = ['BMI', 'DiabetesPedigreeFunction']
    ordinal_list = list(set(data_x.columns) - set(numeric_list))
    parameters = {
        'numeric_cols': numeric_list,
        'ordinal_cols': ordinal_list,
        'categorical_cols': categorical_list,
        'pre_batch_size': 256,
        'pre_epochs': 1000,
        'pre_learning_rate': 1e-3,
        'batch_size': 128,
        'epochs': 500,
        'learning_rate': 1e-4,
        'alpha': 10}

    return parameters

def load_letter_settings(data_x):
    # letter.csv parameters
    ordinal_list = list(data_x.columns)
    categorical_list = []
    numeric_list = []
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
        'alpha': 10}

    return parameters

def load_spam_settings(data_x):
    # spam.csv parameters
    ordinal_list = []
    categorical_list = []
    numeric_list = list(data_x.columns)
    parameters = {
        'numeric_cols': numeric_list,
        'ordinal_cols': ordinal_list,
        'categorical_cols': categorical_list,
        'pre_batch_size': 256,
        'pre_epochs': 20,
        'pre_learning_rate': 1e-4,
        'batch_size': 256,
        'epochs': 100,
        'learning_rate': 1e-4,
        'alpha': 10}

    return parameters

def load_breast_cancer_settings(data_x):
    # breast_cancer.csv parameters
    ordinal_list = []
    categorical_list = []
    numeric_list = list(data_x.columns)
    parameters = {
        'numeric_cols': numeric_list,
        'ordinal_cols': ordinal_list,
        'categorical_cols': categorical_list,
        'pre_batch_size': 256,
        'pre_epochs': 1000,
        'pre_learning_rate': 1e-3,
        'batch_size': 56,
        'epochs': 500,
        'learning_rate': 1e-4,
        'alpha': 10}

    return parameters

def load_credit_card_settings(data_x):
    # credit_card.csv parameters
    numeric_list = []
    categorical_list = ['SEX']
    ordinal_list = list(set(data_x.columns) - set(categorical_list))
    parameters = {
        'numeric_cols': numeric_list,
        'ordinal_cols': ordinal_list,
        'categorical_cols': categorical_list,
        'pre_batch_size': 256,
        'pre_epochs': 200,
        'pre_learning_rate': 1e-3,
        'batch_size': 256,
        'epochs': 50,
        'learning_rate': 1e-4,
        'alpha': 10}

    return parameters