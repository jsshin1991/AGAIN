
# AGAIN
AGAIN: Autoencoder-Generative Adversarial Imputation Network for Missing Data Imputation

We propose a novel generative method for imputing missing data by applying the Conditional Generative Adversarial Network (Conditional GAN) framework. The generator imputes missing components based on the observed components of the missing sample, and the discriminator determines how plausible the imputed missing samples are. To ensure stable training of the generator by balancing the learning speed between the generator and the discriminator, we propose a Conditional GAN structure capable of pre-training and fine-tuning using an autoencoder, and apply it to the missing data imputation. Accordingly, we call our structure Autoencoder-Generative Adversarial Imputation Network (AGAIN). Through the stable training of the generator, our structure ensures that the generator imputes missing samples based on a true data distribution. 

# Keywords
Missing Data Imputation, Generative Adversarial Networks, Autoencoder, Transfer Learning, Deep Learning

# Dataset
All datasets (Wine, Letter, Diabetes, News, Credit Card, Breast Cancer, Spam) used in the experiments can be found in the 'data' directory. (All of them can be found in Kaggle or UCI Machine Learning Dataset.)

**Python Library Version**:
- pandas == 0.25.3
- numpy == 1.14.5
- tensorflow == 1.10.0
- tqdm == 4.46.0
- scikit-learn == 0.23.1
