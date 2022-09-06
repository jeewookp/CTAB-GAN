
# coding: utf-8

# In[1]:


from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D

# from keras.layers.advanced_activations import LeakyReLU @@@@@@@@@@@@@@@@
from keras.layers.activation import LeakyReLU

from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import pandas as pd
import pandas.core.algorithms as algos
import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing


import matplotlib.pyplot as plt

import sys

import numpy as np


# In[2]:


def calculate_performance(predict_dev_result, y_train):
    result_dict = {}
    result_dict['KS'] = 0
    result_dict['AUROC'] = 0
    result_dict['GINI'] = 0
    predict_dev_result_prob = pd.DataFrame(predict_dev_result)
    if predict_dev_result_prob.ix[:,0].unique().shape[0] == 1:
        return result_dict
    predict_dev_result_prob.ix[(predict_dev_result_prob.ix[:, 0] > 0.999), 0] = 0.999
    predict_dev_result_prob.ix[(predict_dev_result_prob.ix[:, 0] < 0.001), 0] = 0.001
    predict_dev_result_prob['logit']= np.log(predict_dev_result_prob.ix[:,0]/(1-predict_dev_result_prob.ix[:,0]))

    predict_dev_result_prob['good'] = y_train
    predict_dev_result_prob['bad'] = 1 - predict_dev_result_prob['good']
    predict_dev_result_prob.columns = ["prob", "logit", 'good', 'bad']

    # bins = algos.quantile(np.unique(predict_dev_result_prob['logit']), np.linspace(0, 1, 21)) @@@@@@@@@@@@@@@@
    bins = pd.DataFrame(np.unique(predict_dev_result_prob['logit'])).quantile(np.linspace(0, 1, 21)).values[:, 0]

    # [2017-06-15] pjh - tools._bins_to_cuts has been removed in pandas version 20.
    predict_dev_result_prob['bucket'] = pd.cut(predict_dev_result_prob['logit'], bins, include_lowest=True)
    grouped = predict_dev_result_prob.groupby('bucket', as_index = False)
    agg1 = pd.DataFrame(grouped.min().prob)
    agg1.columns = ["min_scr"]
    agg1['max_scr'] = grouped.max().prob
    agg1['bads'] = grouped.sum().bad
    agg1['goods'] = grouped.sum().good
    agg1['total'] = agg1.bads + agg1.goods
    agg1['bads_prob'] = agg1['bads'] / agg1['bads'].sum()
    agg1['goods_prob'] = agg1['goods'] / agg1['goods'].sum()

    # agg2 = (agg1.sort_index(by = 'min_scr')).reset_index(drop = True) @@@@@@@@@@@@@@@@
    agg2 = (agg1.sort_values(by = 'min_scr')).reset_index(drop = True)

    agg2['odds'] = (agg2.goods / agg2.bads).apply('{0:.2f}'.format)
    agg2['bad_rate'] = (agg2.bads / agg2.total).apply('{0:.2%}'.format)
    agg2['ks'] = abs(np.round(((agg2.bads / predict_dev_result_prob.bad.sum()).cumsum() - (agg2.goods / predict_dev_result_prob.good.sum()).cumsum()), 4))
    flag = lambda x: '<----' if x == agg2.ks.max() else ''
    agg2['max_ks'] = agg2.ks.apply(flag)
    auroc = (( agg2['goods_prob'] * agg2['bads_prob'] / 2.0) + (1 - agg2['goods_prob'].cumsum()) * agg2['bads_prob']).sum()
    # To prevent GINI from being negative
    if auroc < 0.5:
        auroc = 1 - auroc

    # the performances should be 'float' type(one of primitive type)
    # result_dict['KS'] = agg2.ks.max().item() @@@@@@@@@@@@@@@@
    result_dict['KS'] = agg2.ks.max()
    result_dict['AUROC'] = auroc
    result_dict['GINI'] = auroc * 2.0 - 1.0
    return result_dict


# In[3]:


# df_train = pd.read_csv('kfb_sam_ml_dev.txt', delimiter="|") @@@@@@@@@@@@@@@@
# df_test = pd.read_csv('kfb_sam_ml_val.txt', delimiter="|") @@@@@@@@@@@@@@@@
df_train = pd.read_csv('kfb_sam_ml_dev.txt')
df_test = pd.read_csv('kfb_sam_ml_val.txt')



# df_train = df_train.reset_index(drop=True) @@@@@@@@@@@@@@@@
# df_test = df_test.reset_index(drop=True) @@@@@@@@@@@@@@@@
# df_train.describe() @@@@@@@@@@@@@@@@


# In[4]:


x_train = df_train.drop(['dev_val', 'pk','bad'], axis=1).values
x_test = df_test.drop(['dev_val', 'pk','bad'], axis=1).values

std_scalar = preprocessing.MinMaxScaler()
x_train = std_scalar.fit_transform(x_train)
x_test = std_scalar.transform(x_test)

Y_train = pd.get_dummies(df_train.bad).values
Y_test = pd.get_dummies(df_test.bad).values

x_train_gan = 2 * np.concatenate([x_train, Y_train], axis=1) - 1
x_test_gan = 2 * np.concatenate([x_test, Y_test], axis=1) - 1


# In[29]:


# 원래 데이터를 활용한 DNN Performance 
# Neural Net for Collection Reactions
original_model = Sequential()
original_model.add(Dense(units=50, input_dim=x_train.shape[1], activation='relu', kernel_initializer='he_normal'))
original_model.add(Dropout(0.2))
original_model.add(BatchNormalization())
original_model.add(Dense(units=50, activation='relu', kernel_initializer='he_normal'))
original_model.add(Dropout(0.2))
original_model.add(BatchNormalization())
original_model.add(Dense(units=2, activation='softmax'))
original_model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001))
# origninal_result = original_model.model.fit(x_train, Y_train, epochs=50, batch_size=64, class_weight='auto', verbose=2) @@@@@@@@@@@@@@@@
origninal_result = original_model.fit(x_train, Y_train, epochs=50, batch_size=64, verbose=2)


# In[34]:


# y_result = original_model.predict_proba(x_test)[:, :1] # predict @@@@@@@@@@@@@@@@
y_result = original_model.predict_on_batch(x_test)[:, :1]

perf = calculate_performance(y_result, df_test.bad.values) # Calculate Performance
perf
# {'KS': 0.2744, 'AUROC': 0.6874410759357457, 'GINI': 0.3748821518714913}


# In[5]:


class GAN():
    def __init__(self):
        self.dat_cols = x_train_gan.shape[1]
        self.channels = 1
        self.dat_shape = self.dat_cols
        self.latent_dim = 10

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        fakedata = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(fakedata)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(10, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(15))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(20))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.dat_shape, activation='tanh'))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        dat = model(noise)

        return Model(noise, dat)

    def build_discriminator(self):

        model = Sequential()
        model.add(Dense(20, input_dim=self.dat_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))
        model.add(Dense(10))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        dat = Input(shape=(self.dat_shape,))
        validity = model(dat)

        return Model(dat, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
#         (X_train, _), (_, _) = mnist.load_data()
        X_train = x_train_gan

        # Rescale -1 to 1
#         X_train = X_train / 127.5 - 1.
#         X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            dats = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_dats = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(dats, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_dats, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)
#             g_loss = self.combined.train_on_batch(noise, fake)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
#             if epoch % sample_interval == 0:
#                 self.sample_datas(epoch)

    def sample_datas(self, epoch):
        r, c = 1, self.dat_cols
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_dats = self.generator.predict(noise)
        gen_dats = (gen_dats + 1) / 2
        # Rescale & invers transform
        inv_gen_dats = np.concatenate([std_scalar.inverse_transform(gen_dats[:,:22]), gen_dats[:,22:]], axis=1)
        print(inv_gen_dats)


# In[7]:


# if __name__ == '__main__':
# gan = GAN()
gan.train(epochs=300, batch_size=1024, sample_interval=200)


# In[9]:


# fake data generate

r = x_train_gan.shape[0]
noise = np.random.normal(0, 1, (r,  10))
gen_dats = gan.generator.predict(noise)
gen_dats = (gen_dats + 1) / 2
# Rescale & invers transform
x_train_fake = gen_dats[:,:22]
Y_train_fake = gen_dats[:,22:]


# In[10]:


# 원래 데이터를 활용한 DNN Performance 
# Neural Net for Collection Reactions
fake_model = Sequential()
fake_model.add(Dense(units=50, input_dim=x_train.shape[1], activation='relu', kernel_initializer='he_normal'))
fake_model.add(Dropout(0.2))
fake_model.add(BatchNormalization())
fake_model.add(Dense(units=50, activation='relu', kernel_initializer='he_normal'))
fake_model.add(Dropout(0.2))
fake_model.add(BatchNormalization())
fake_model.add(Dense(units=2, activation='softmax'))
fake_model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001))
fake_result = fake_model.model.fit(x_train_fake, Y_train_fake, epochs=10, batch_size=64, class_weight='auto', verbose=2)


# In[11]:


y_result = fake_model.predict_proba(x_test)[:, :1] # predict
perf = calculate_performance(y_result, df_test.bad.values) # Calculate Performance
perf
# {'KS': 0.1006, 'AUROC': 0.5619874890772766, 'GINI': 0.12397497815455316}
# {'KS': 0.1042, 'AUROC': 0.5586508520701625, 'GINI': 0.11730170414032504}
# {'KS': 0.1713, 'AUROC': 0.5949583505542383, 'GINI': 0.18991670110847658}


# In[239]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


# In[240]:


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# In[320]:


X_train = x_train_gan
X_test = x_test_gan
column_size = x_train_gan.shape[1]

# network parameters
input_shape = (column_size, )
batch_size = 128
latent_dim = 50
epochs = 50
intermediate_dim = 30
original_dim = column_size
# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation=LeakyReLU(alpha=0.2))(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation=LeakyReLU(alpha=0.2))(latent_inputs)
outputs = Dense(original_dim, activation='tanh')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')
models = (encoder, decoder)
# reconstruction_loss = binary_crossentropy(inputs,outputs)    
reconstruction_loss = mse(inputs,outputs)    
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam', loss=None)
vae.summary()

# train the autoencoder
vae.fit(x_train_gan,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test_gan, None))
vae.save_weights('vae_mlp_mnist.h5')


# In[324]:


# train the autoencoder
vae.fit(x_train_gan,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test_gan, None))
vae.save_weights('vae_mlp_mnist.h5')


# In[321]:


r = x_train_gan.shape[0]
latent_x = np.random.normal(0, 1, (r,  50))

gen_dats = decoder.predict(latent_x)
gen_dats = (gen_dats + 1) / 2
# Rescale & invers transform
x_train_fake = gen_dats[:,:22]
Y_train_fake = gen_dats[:,22:]


# In[322]:


# 원래 데이터를 활용한 DNN Performance 
# Neural Net for Collection Reactions
fake_model = Sequential()
fake_model.add(Dense(units=50, input_dim=x_train.shape[1], activation='relu', kernel_initializer='he_normal'))
fake_model.add(Dropout(0.2))
fake_model.add(BatchNormalization())
fake_model.add(Dense(units=50, activation='relu', kernel_initializer='he_normal'))
fake_model.add(Dropout(0.2))
fake_model.add(BatchNormalization())
fake_model.add(Dense(units=2, activation='softmax'))
fake_model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001))
fake_result = fake_model.model.fit(x_train_fake, Y_train_fake, epochs=10, batch_size=64, class_weight='auto', verbose=2)


# In[323]:


y_result = fake_model.predict_proba(x_test)[:, :1] # predict
perf = calculate_performance(y_result, df_test.bad.values) # Calculate Performance
perf


# In[325]:


from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D

# from keras.layers.advanced_activations import LeakyReLU @@@@@@@@@@@@@@@@
from keras.layers.activation import LeakyReLU

from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np


# In[410]:


class SGAN:
    def __init__(self):
        self.dat_cols = x_train_gan.shape[1] - 2
        self.channels = 1
        self.dat_shape = self.dat_cols
        self.num_classes = 2
        self.latent_dim = 10

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss=['binary_crossentropy', 'categorical_crossentropy'],
            loss_weights=[0.5, 0.5],
            optimizer=optimizer,
            metrics=['accuracy']
        )

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        noise = Input(shape=(self.latent_dim,))
        fakedata = self.generator(noise)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid, _ = self.discriminator(fakedata)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(noise, valid)
        self.combined.compile(loss=['binary_crossentropy'], optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(10, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(15))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(20))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.dat_shape, activation='tanh'))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model = Sequential()
        model.add(Dense(20, input_dim=self.dat_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))
        model.add(Dense(10))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        dat = Input(shape=(self.dat_shape,))

        features = model(dat)
        valid = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes+1, activation="softmax")(features)

        return Model(dat, [valid, label])

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
#         (X_train, y_train), (_, _) = mnist.load_data()
#         X_train = x_train_gan
        X_train = x_train_gan[:,:22]

        # Rescale -1 to 1
#         X_train = (X_train.astype(np.float32) - 127.5) / 127.5
#         X_train = np.expand_dims(X_train, axis=3)
#         y_train = y_train.reshape(-1, 1)
        y_train = df_train.bad.values
        # Class weights:
        # To balance the difference in occurences of digit class labels.
        # 50% of labels that the discriminator trains on are 'fake'.
        # Weight = 1 / frequency
        half_batch = batch_size // 2
        cw1 = {0: 1, 1: 1}
        cw2 = {i: self.num_classes / half_batch for i in range(self.num_classes)}
        cw2[self.num_classes] = 1 / half_batch

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            dats = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_dats = self.generator.predict(noise)

            # One-hot encoding of labels
            labels = to_categorical(y_train[idx], num_classes=self.num_classes+1)
            fake_labels = to_categorical(np.full((batch_size, 1), self.num_classes), num_classes=self.num_classes+1)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(dats, [valid, labels], class_weight=[cw1, cw2])
            d_loss_fake = self.discriminator.train_on_batch(gen_dats, [fake, fake_labels], class_weight=[cw1, cw2])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid, class_weight=[cw1, cw2])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss))

  
    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "sgan_generator")
        save(self.discriminator, "sgan_discriminator")
        save(self.combined, "sgan_adversarial")


# In[420]:


# sgan = SGAN()
sgan.train(epochs=300000, batch_size=1024, sample_interval=200)


# In[421]:


# fake data generate
def sum_col(x):
    x1 = x[0] / (x[0]+ x[1])
    x2 = x[1] / (x[0]+ x[1])
    return x1, x2
#     if x[0] > x[1]:
#         return 1, 0
#     else:
#         return 0, 1


r = x_train_gan.shape[0]
noise = np.random.normal(0, 1, (r,  10))
gen_dats = sgan.generator.predict(noise)
_, Y_train_fake = sgan.discriminator.predict(gen_dats)
Y_train_fake = list(map(sum_col, Y_train_fake[:,1:3]))
# Rescale & invers transform
x_train_fake = (gen_dats + 1) / 2


# In[422]:


# 원래 데이터를 활용한 DNN Performance 
# Neural Net for Collection Reactions
fake_model = Sequential()
fake_model.add(Dense(units=50, input_dim=x_train.shape[1], activation='relu', kernel_initializer='he_normal'))
fake_model.add(Dropout(0.2))
fake_model.add(BatchNormalization())
fake_model.add(Dense(units=50, activation='relu', kernel_initializer='he_normal'))
fake_model.add(Dropout(0.2))
fake_model.add(BatchNormalization())
fake_model.add(Dense(units=2, activation='softmax'))
fake_model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001))
fake_result = fake_model.model.fit(x_train_fake, Y_train_fake, epochs=10, batch_size=64, class_weight='auto', verbose=2)


# In[423]:


y_result = fake_model.predict_proba(x_test)[:, :1] # predict
perf = calculate_performance(y_result, df_test.bad.values) # Calculate Performance
perf

