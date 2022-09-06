import pandas as pd
import numpy as np
from sklearn import preprocessing
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, BatchNormalization, Lambda
from keras.optimizers import Adam
from keras.layers.activation import LeakyReLU
from keras import backend as K
from keras.losses import mse

def calculate_performance(predict_dev_result, y_train):
    result_dict = {}
    result_dict['KS'] = 0
    result_dict['AUROC'] = 0
    result_dict['GINI'] = 0
    predict_dev_result_prob = pd.DataFrame(predict_dev_result)
    if predict_dev_result_prob.iloc[:,0].unique().shape[0] == 1:
        return result_dict
    predict_dev_result_prob.iloc[(predict_dev_result_prob.iloc[:, 0] > 0.999), 0] = 0.999
    predict_dev_result_prob.iloc[(predict_dev_result_prob.iloc[:, 0] < 0.001), 0] = 0.001
    predict_dev_result_prob['logit']= np.log(predict_dev_result_prob.iloc[:,0]/(1-predict_dev_result_prob.iloc[:,0]))

    predict_dev_result_prob['good'] = y_train
    predict_dev_result_prob['bad'] = 1 - predict_dev_result_prob['good']
    predict_dev_result_prob.columns = ["prob", "logit", 'good', 'bad']

    bins = pd.DataFrame(np.unique(predict_dev_result_prob['logit'])).quantile(np.linspace(0, 1, 21)).values[:, 0]

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

    agg2 = (agg1.sort_values(by = 'min_scr')).reset_index(drop = True)

    agg2['odds'] = (agg2.goods / agg2.bads).apply('{0:.2f}'.format)
    agg2['bad_rate'] = (agg2.bads / agg2.total).apply('{0:.2%}'.format)
    agg2['ks'] = abs(np.round(((agg2.bads / predict_dev_result_prob.bad.sum()).cumsum() - (agg2.goods / predict_dev_result_prob.good.sum()).cumsum()), 4))
    flag = lambda x: '<----' if x == agg2.ks.max() else ''
    agg2['max_ks'] = agg2.ks.apply(flag)
    auroc = (( agg2['goods_prob'] * agg2['bads_prob'] / 2.0) + (1 - agg2['goods_prob'].cumsum()) * agg2['bads_prob']).sum()
    if auroc < 0.5:
        auroc = 1 - auroc

    result_dict['KS'] = agg2.ks.max()
    result_dict['AUROC'] = auroc
    result_dict['GINI'] = auroc * 2.0 - 1.0
    return result_dict

class GAN():
    def __init__(self):
        self.dat_cols = x_train_gan.shape[1]
        self.channels = 1
        self.dat_shape = self.dat_cols
        self.latent_dim = 10

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dim,))
        fakedata = self.generator(z)

        self.discriminator.trainable = False

        validity = self.discriminator(fakedata)

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

    def train(self, epochs, batch_size, sample_interval, print_interval):

        X_train = x_train_gan

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            idx = np.random.randint(0, X_train.shape[0], batch_size)
            dats = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            gen_dats = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(dats, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_dats, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            g_loss = self.combined.train_on_batch(noise, valid)
            if epoch%print_interval==0:
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % sample_interval == 0:
                self.sample_datas(epoch)

    def sample_datas(self, epoch):
        noise = np.random.normal(0, 1, (self.dat_cols, self.latent_dim))
        gen_dats = self.generator.predict(noise)
        gen_dats = (gen_dats + 1) / 2
        inv_gen_dats = np.concatenate([std_scalar.inverse_transform(gen_dats[:,:22]), gen_dats[:,22:]], axis=1)
        print(inv_gen_dats)

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


df_train = pd.read_csv('/Users/john/data/ML_data_dev.csv')
df_test = pd.read_csv('/Users/john/data/ML_data_val.csv')

x_train = df_train.drop(['dev_val', 'pk','bad'], axis=1).values
x_test = df_test.drop(['dev_val', 'pk','bad'], axis=1).values

std_scalar = preprocessing.MinMaxScaler()
x_train = std_scalar.fit_transform(x_train)
x_test = std_scalar.transform(x_test)

Y_train = pd.get_dummies(df_train.bad).values
Y_test = pd.get_dummies(df_test.bad).values

x_train_gan = 2 * np.concatenate([x_train, Y_train], axis=1) - 1
x_test_gan = 2 * np.concatenate([x_test, Y_test], axis=1) - 1

original_model = Sequential()
original_model.add(Dense(units=50, input_dim=x_train.shape[1], activation='relu', kernel_initializer='he_normal'))
original_model.add(Dropout(0.2))
original_model.add(BatchNormalization())
original_model.add(Dense(units=50, activation='relu', kernel_initializer='he_normal'))
original_model.add(Dropout(0.2))
original_model.add(BatchNormalization())
original_model.add(Dense(units=2, activation='softmax'))
original_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))

origninal_result = original_model.fit(x_train, Y_train, epochs=50, batch_size=64, verbose=2)
y_result = original_model.predict_on_batch(x_test)[:, :1]
perf = calculate_performance(y_result, df_test.bad.values)
print(perf)


gan = GAN()
gan.train(epochs=300, batch_size=1024, sample_interval=200,print_interval=100)

r = x_train_gan.shape[0]
noise = np.random.normal(0, 1, (r,  10))
gen_dats = gan.generator.predict(noise)
gen_dats = (gen_dats + 1) / 2
x_train_fake = gen_dats[:,:22]
Y_train_fake = gen_dats[:,22:]

fake_model = Sequential()
fake_model.add(Dense(units=50, input_dim=x_train.shape[1], activation='relu', kernel_initializer='he_normal'))
fake_model.add(Dropout(0.2))
fake_model.add(BatchNormalization())
fake_model.add(Dense(units=50, activation='relu', kernel_initializer='he_normal'))
fake_model.add(Dropout(0.2))
fake_model.add(BatchNormalization())
fake_model.add(Dense(units=2, activation='softmax'))
fake_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001))

fake_result = fake_model.fit(x_train_fake, Y_train_fake, epochs=10, batch_size=64, verbose=2)
y_result = fake_model.predict_on_batch(x_test)[:, :1]
perf = calculate_performance(y_result, df_test.bad.values)
print(perf)


X_train = x_train_gan
X_test = x_test_gan
column_size = x_train_gan.shape[1]

input_shape = (column_size, )
batch_size = 128
latent_dim = 50
epochs = 50
intermediate_dim = 30
original_dim = column_size

inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation=LeakyReLU(alpha=0.2))(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation=LeakyReLU(alpha=0.2))(latent_inputs)
outputs = Dense(original_dim, activation='tanh')(x)

decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')
reconstruction_loss = mse(inputs,outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam', loss=None)
vae.summary()

vae.fit(x_train_gan,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test_gan, None))

r = x_train_gan.shape[0]
latent_x = np.random.normal(0, 1, (r,  50))

gen_dats = decoder.predict(latent_x)
gen_dats = (gen_dats + 1) / 2
x_train_fake = gen_dats[:,:22]
Y_train_fake = gen_dats[:,22:]

fake_model = Sequential()
fake_model.add(Dense(units=50, input_dim=x_train.shape[1], activation='relu', kernel_initializer='he_normal'))
fake_model.add(Dropout(0.2))
fake_model.add(BatchNormalization())
fake_model.add(Dense(units=50, activation='relu', kernel_initializer='he_normal'))
fake_model.add(Dropout(0.2))
fake_model.add(BatchNormalization())
fake_model.add(Dense(units=2, activation='softmax'))
fake_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001))

fake_result = fake_model.fit(x_train_fake, Y_train_fake, epochs=10, batch_size=64, verbose=2)
y_result = fake_model.predict_on_batch(x_test)[:, :1]
perf = calculate_performance(y_result, df_test.bad.values)
print(perf)