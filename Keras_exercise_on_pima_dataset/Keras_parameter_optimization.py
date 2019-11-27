import numpy as np
from numpy import loadtxt
# import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import layers
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

data = loadtxt(r"C:\Users\Sonya\Desktop\code\pima.csv", delimiter=',')
np.random.shuffle(data)
split_percent = 0.8
train, test = np.split(data, [int(split_percent * len(data))])
x_train = train[:, 0:-1]
y_train = train[:, -1]
x_test = test[:, 0:-1]
y_test = test[:, -1]


def my_model():
    model = keras.Sequential()
    model.add(layers.Dense(12, input_dim=8, activation="relu", kernel_initializer='uniform'))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
    return model


model = KerasClassifier(build_fn=my_model)
batch = list(np.arange(10, 100, 20))
epochs = [10, 50, 100]
batch_epoch = dict(batch_size=batch, epochs=epochs)
grid_batch_epoch = GridSearchCV(estimator=model, param_grid=grid_param_batch_epoch, n_jobs=-1, cv=3)
results_grid_batch = grid_batch_epoch.fit(x_train, y_train)

def my_model_1(optimizer='adam'):
    model = keras.Sequential()
    model.add(layers.Dense(12, input_dim=8, activation="relu", kernel_initializer='uniform'))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


model1 = KerasClassifier(build_fn=my_model_1, batch_size=10, epochs=100)
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizer)
grid_optimizer = GridSearchCV(estimator=model1, param_grid=param_grid, n_jobs=-1, cv=3)
results_optimizer = grid_optimizer.fit(x_train, y_train)


######################################################################################################################
def my_model_2(learn_rate=0.1, momentum=0):
    model = keras.Sequential()
    model.add(layers.Dense(12, input_dim=8, activation="relu", kernel_initializer='uniform'))
    model.add(layers.Dense(1, activation="sigmoid"))
    optimizer = keras.optimizers.SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


model_2 = KerasClassifier(build_fn=my_model_2, batch_size=10, epochs=100)
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
lr_mom_grid = dict(learn_rate=learn_rate, momentum=momentum)
grid_lr_momentum = GridSearchCV(estimator=model_2, param_grid=lr_mom_grid, n_jobs=-1, cv=3)
results_lr_mom = grid_lr_momentum.fit(x_train, y_train)


########################################################################################################################

def my_model_3(initializer='uniform'):
    model = keras.Sequential()
    model.add(layers.Dense(12, input_dim=8, activation="relu", kernel_initializer=initializer))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer='Nadam', metrics=["accuracy"])
    return model


model_3 = KerasClassifier(build_fn=my_model_3, batch_size=10, epochs=100)
initializer = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal',
               'he_uniform']
init_grid_param = dict(initializer=initializer)
grid_init=GridSearchCV(estimator=model_3,param_grid=init_grid_param,n_jobs=-1, cv=3)
results_init=grid_init.fit(x_train,y_train)

########################################################################################################################

def my_model_4(act="relu"):
    model = keras.Sequential()
    model.add(layers.Dense(12, input_dim=8, activation=act, kernel_initializer='glorot_normal'))
    model.add(layers.Dense(1, activation="sigmoid", kernel_initializer='glorot_normal'))
    model.compile(loss="binary_crossentropy", optimizer='Nadam', metrics=["accuracy"])
    return model



model_4 = KerasClassifier(build_fn=my_model_4, batch_size=10, epochs=100)
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
act_param = dict(act=activation)
grid_act = GridSearchCV(estimator=model_4, param_grid=act_param, n_jobs=-1, cv=3)
results_act = grid_act.fit(x_train, y_train)


#######################################################################################################################

def my_model_5(drop_rate=0.1, w_constrains=0):
    model = keras.Sequential()
    model.add(layers.Dense(12, input_dim=8, activation='softplus', kernel_initializer='glorot_normal',
                           kernel_constraint=keras.constraints.maxnorm(w_constrains)))
    model.add(layers.Dropout(rate=drop_rate))
    model.add(layers.Dense(1, activation="sigmoid", kernel_initializer='glorot_normal'))
    model.compile(loss="binary_crossentropy", optimizer='Nadam', metrics=["accuracy"])
    return model


model_5 = KerasClassifier(build_fn=my_model_5, batch_size=10, epochs=100)
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
grid_drop_param = dict(drop_rate=dropout_rate, w_constrains=weight_constraint)
grid_drop = GridSearchCV(estimator=model_5, param_grid=grid_drop_param, n_jobs=-1, cv=3)
results_drop = grid_drop.fit(x_train, y_train)


###################################################################################################################
def my_model_6(num_neurons):
    model = keras.Sequential()
    model.add(layers.Dense(num_neurons, input_dim=8, activation='softplus', kernel_initializer='glorot_normal',
                           kernel_constraint=keras.constraints.maxnorm(2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation="sigmoid", kernel_initializer='glorot_normal'))
    model.compile(loss="binary_crossentropy", optimizer='Nadam', metrics=["accuracy"])
    return model


model_6 = KerasClassifier(build_fn=my_model_6, batch_size=10, epochs=100)
neurons = [1, 5, 10, 15, 20, 25, 30]
neuron_grid_param = dict(num_neurons=neurons)
neurons_grid = GridSearchCV(estimator=model_6, param_grid=neuron_grid_param, n_jobs=-1, cv=3)
results_neurons = neurons_grid.fit(x_train, y_train)


#######################################################################################################################
def final_model():
    model = keras.Sequential()
    model.add(layers.Dense(20, input_dim=8, activation='softplus', kernel_initializer='glorot_normal',
                           kernel_constraint=keras.constraints.maxnorm(2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation="sigmoid", kernel_initializer='glorot_normal'))
    model.compile(loss="binary_crossentropy", optimizer='Nadam', metrics=["accuracy"])
    return model


model = final_model()
model.fit(x_train, y_train, batch_size=10, epochs=200)
accuracy = model.evaluate(x_train, y_train)
test = model.predict(x_test)
print(f'Accuracy is: {accuracy[1] * 100:.2f} %')
