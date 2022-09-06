import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import keras as tf_keras
from sklearn.preprocessing import StandardScaler
import sklearn
import importlib_metadata
from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import class_weight

def ks_stat(y, yhat):
    return ks_2samp(yhat[y==1], yhat[y!=1]).statistic

def get_initializer(activation_func):
    if activation_func == 'sigmoid' or activation_func == 'tanh':
        return 'glorot_normal'
    elif activation_func == 'relu' or activation_func == 'elu' or activation_func == 'selu':
        return 'he_normal'
    else:
        return 'glorot_normal'

def get_optimizer(optimizer_func_name, learning_rate):
    opt = None
    if optimizer_func_name == 'SGD':
        opt = tf_keras.optimizers.SGD(lr=learning_rate)
    elif optimizer_func_name == 'RMSprop':
        opt = tf_keras.optimizers.RMSprop(lr=learning_rate)
    elif optimizer_func_name == 'Adagrad':
        opt = tf_keras.optimizers.Adagrad(lr=learning_rate)
    elif optimizer_func_name == 'Adadelta':
        opt = tf_keras.optimizers.Adadelta(lr=learning_rate)
    elif optimizer_func_name == 'Adam':
        opt = tf_keras.optimizers.Adam(lr=learning_rate)
    elif optimizer_func_name == 'Adamax':
        opt = tf_keras.optimizers.Adamax(lr=learning_rate)
    else:
        opt = tf_keras.optimizers.Adam(lr=learning_rate)
    return opt

def get_output_activation_func(is_target_discrete):
    return 'softmax' if is_target_discrete else 'linear'

df_dev = pd.read_csv('/Users/john/data/ML_data_dev.csv')
df_val = pd.read_csv('/Users/john/data/ML_data_val.csv')

dev_df_x = df_dev.drop(['dev_val', 'pk','bad'], axis=1)
val_df_x = df_val.drop(['dev_val', 'pk','bad'], axis=1)

dev_df_y = pd.get_dummies(df_dev.bad)
val_df_y = pd.get_dummies(df_val.bad)

x_train = dev_df_x.values
y_train = dev_df_y.values
x_val = val_df_x.values
y_val = val_df_y.values

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

activation_func = 'relu'
l1_penalty = 0.00001
l2_penalty = 0.00001
input_dropout_ratio = 0.1
is_target_discrete = True
hidden_layers_info = [50, 50, 50]
OutputDropoutRatio = [0.15, 0.15, 0.15]
kernel_regularizer = tf_keras.regularizers.l1_l2(l1=l1_penalty, l2=l2_penalty)
kernel_initializer = get_initializer(activation_func)
output_activation = get_output_activation_func(is_target_discrete)
num_input_nodes = x_train.shape[1]
use_batch_normalization = True
num_output_layers = 2

model = tf_keras.models.Sequential()

if len(hidden_layers_info) > 0:
    # input dropout ratio
    if input_dropout_ratio > 0.0:
        model.add(tf_keras.layers.Dropout(input_dropout_ratio, input_shape=(num_input_nodes,),
                            name='input_layer'))
    ## Hidden layers
    for i in range(0, len(hidden_layers_info)):
        if (i == 0 and input_dropout_ratio > 0.0) or (i > 0):
            model.add(tf_keras.layers.Dense(units=hidden_layers_info[i],
                            kernel_regularizer=kernel_regularizer,
                            kernel_initializer=kernel_initializer,
                            name='hidden' + str(i)))
        else:
            model.add(tf_keras.layers.Dense(units=hidden_layers_info[i],
                            input_shape=(num_input_nodes,),
                            kernel_regularizer=kernel_regularizer,
                            kernel_initializer=kernel_initializer,
                            name='hidden' + str(i)))
        if use_batch_normalization:
            model.add(tf_keras.layers.BatchNormalization())
        model.add(tf_keras.layers.Activation(activation_func))
        if OutputDropoutRatio[i] > 0.0:
            model.add(tf_keras.layers.Dropout(OutputDropoutRatio[i]))

    model.add(tf_keras.layers.Dense(units=num_output_layers, name='output_layer', activation=output_activation))
else:
    model.add(tf_keras.layers.Dense(units=num_output_layers,  input_shape=(num_input_nodes, ), activation=output_activation,
                                    name='output_layer'))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy'])

class_weights = dict(zip(np.unique(y_train[:,0]), class_weight.compute_class_weight(
                class_weight = 'balanced', classes=np.unique(y_train[:,0]), y =y_train[:,0])))
n_epochs = 5
batch_size = 1024
has_validation_set = True

result = model.fit(x_train, y_train,
                epochs=n_epochs, batch_size=batch_size, class_weight= class_weights,
                validation_data=(x_val, y_val) if has_validation_set else None, verbose=2)

pred_dev = model.predict(x_train)
pred_val = model.predict(x_val)
ks_dev = ks_stat(df_dev.bad, pred_dev[:,1])
ks_val = ks_stat(df_val.bad, pred_val[:,1])
print(ks_dev)
print(ks_val)

n_epochs = 100
batch_size = 1024
result = model.fit(x_val, y_val,
                        epochs=n_epochs, batch_size=batch_size, class_weight= class_weights,
                        validation_split= 0.3, verbose=2)

pred_dev = model.predict(x_train)
pred_val = model.predict(x_val)
ks_dev = ks_stat(df_dev.bad, pred_dev[:,1])
ks_val = ks_stat(df_val.bad, pred_val[:,1])

print(ks_dev)
print(ks_val)


def evaluate(model, test_set):
  # acc = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
  acc = tf.keras.metrics.BinaryAccuracy(name='accuracy')
  for i, (imgs, labels) in enumerate(test_set):
    preds = model.predict_on_batch(imgs)
    acc.update_state(labels, preds)
  return acc.result().numpy()

def compute_precision_matrices(model, task_set, num_batches=1, batch_size=32):
  task_set = task_set.repeat()
  precision_matrices = {n: tf.zeros_like(p.value()) for n, p in enumerate(model.trainable_variables)}

  for i, (imgs, labels) in enumerate(task_set.take(num_batches)):
    # We need gradients of model params
    with tf.GradientTape() as tape:
      # Get model predictions for each image
      preds = model(imgs)
      # Get the log likelihoods of the predictions
      ll = tf.nn.log_softmax(preds)
    # Attach gradients of ll to ll_grads
    ll_grads = tape.gradient(ll, model.trainable_variables)
    # Compute F_i as mean of gradients squared
    for i, g in enumerate(ll_grads):
      precision_matrices[i] += tf.math.reduce_mean(g ** 2, axis=0) / num_batches

  return precision_matrices

def compute_elastic_penalty(F, theta, theta_A, alpha=1):
  penalty = 0
  for i, theta_i in enumerate(theta):
    _penalty = tf.math.reduce_sum(F[i] * (theta_i - theta_A[i]) ** 2)
    penalty += _penalty
  return 0.5*alpha*penalty

def ewc_loss(labels, preds, model, F, theta_A):
  loss_b = model.loss(labels, preds)
  penalty = compute_elastic_penalty(F, model.trainable_variables, theta_A)
  return loss_b + penalty

def train_with_ewc(model, task_A_set, task_B_set, epochs=3):
  theta_A = {n: p.value() for n, p in enumerate(model.trainable_variables.copy())}
  F = compute_precision_matrices(model, task_A_set, num_batches=1000)

  print("Task A accuracy after training on Task A: {}".format(evaluate(model, task_A_set)))
  print("Task B accuracy after training on Task A: {}".format(evaluate(model, task_B_set)))

  # Now we set up the training loop for task B with EWC
  accuracy = tf.keras.metrics.BinaryAccuracy('accuracy')
  loss = tf.keras.metrics.BinaryCrossentropy('loss')

  # for epoch in range(epochs*3):
  for epoch in range(epochs):
    accuracy.reset_states()
    loss.reset_states()

    for batch, (imgs, labels) in enumerate(task_B_set):
      with tf.GradientTape() as tape:
        # Make the predictions
        preds = model(imgs)
        # Compute EWC loss
        total_loss = ewc_loss(labels, preds, model, F, theta_A)
      # Compute the gradients of model's trainable parameters wrt total loss
      grads = tape.gradient(total_loss, model.trainable_variables)
      # Update the model with gradients
      model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
      # Report updated loss and accuracy
      accuracy.update_state(labels, preds)
      loss.update_state(labels, preds)
      print("\rEpoch: {}, Batch: {}, Loss: {:.3f}, Accuracy: {:.3f}".format(
          epoch+1, batch+1, loss.result().numpy(), accuracy.result().numpy()), flush=True, end=''
         )
    print("")
  print("Task A accuracy after training trained model on Task B: {}".format(evaluate(model, task_A_set)))
  print("Task B accuracy after training trained model on Task B: {}".format(evaluate(model, task_B_set)))

task_A_set = tf.data.Dataset.from_tensor_slices((x_train.astype(np.float32), y_train.astype(np.float32)))\
    .shuffle(len(x_train)).batch(1024)
task_B_set = tf.data.Dataset.from_tensor_slices((x_val.astype(np.float32), y_val.astype(np.float32)))\
    .shuffle(len(x_val)).batch(1024)

model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy'])
model.fit(task_A_set, epochs=100)

pred_dev = model.predict(x_train)
pred_val = model.predict(x_val)
ks_dev = ks_stat(df_dev.bad, pred_dev[:,1])
ks_val = ks_stat(df_val.bad, pred_val[:,1])

print(ks_dev)
print(ks_val)

train_with_ewc(model, task_A_set, task_B_set, epochs=100)

pred_dev = model.predict(x_train)
pred_val = model.predict(x_val)
ks_dev = ks_stat(df_dev.bad, pred_dev[:,1])
ks_val = ks_stat(df_val.bad, pred_val[:,1])

print(ks_dev)
print(ks_val)














