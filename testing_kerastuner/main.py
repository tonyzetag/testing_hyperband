import os
import cv2
import math
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, RandomFlip, RandomRotation, RandomZoom, Rescaling, BatchNormalization, Activation
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.data import AUTOTUNE
from sklearn.model_selection import train_test_split
import keras_tuner as kt
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import matplotlib as mpl
import graphviz
import pydot

class HyperModel(kt.HyperModel):
    def __init__(self, input_shape, output_size):
        self.input_shape = input_shape
        self.output_size = output_size
        
    def build(self, hp):
        # Hyperparameters - Data augmentation
        hp_factor = hp.Float('factor', min_value=0.05, max_value=0.15)
        hp_height_factor = hp.Float('height_factor', min_value=0.05, max_value=0.15)
        # Hyperparameters - CNN
        hp_n_layers = hp.Int('n_layers', min_value=3, max_value=5, step=1)
        hp_filters = hp.Int('filters', min_value=8, max_value=32, step=8) # * (n + 1)
        hp_kernel_size = hp.Choice('kernel_size', [3, 5])
        hp_use_batch_normalization = hp.Choice('use_batch_normalization', [True, False])
        # Hyperparameters - Compile
        hp_lr = hp.Float("lr", min_value=1e-5, max_value=1e-3, sampling="log")
        # Hyperparameters - pool_size
        hp_pool_size = hp.Choice('pool_size', [2, 3])
        # Modelo Sequencial
        model = Sequential()
        # Data augmentation
        model.add(RandomFlip('horizontal', input_shape=input_shape))
        model.add(RandomRotation(hp_factor))
        model.add(RandomZoom(hp_height_factor))
        # Escalado (normalizado)
        model.add(Rescaling(1./255))
        # CNN
        for n in range(2):
            model.add(Conv2D(filters=hp_filters*(2**n), kernel_size=hp_kernel_size, padding='same'))
            if hp_use_batch_normalization:
                model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=2))
        for n in range(2, hp_n_layers):
            model.add(Conv2D(filters=hp_filters*(2**n), kernel_size=3, padding='same'))
            if hp_use_batch_normalization:
                model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=hp_pool_size))
        model.add(Flatten())
        model.add(Dense(self.output_size))
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
        return model

seed = 36645
batch_size = 32
image_size=(256, 256)
channels = 3
input_shape = (image_size[0], image_size[1], channels)
epochs = 100
hyperband_max_epochs = 50
factor = 40 # Factor de reducción, mas alto, menos iteraciones

# From kaggle
flowers_path = 'D:\\Datasets\\flowers\\flowers'

# Labels
labels = os.listdir(flowers_path)

# Usamos la utilidad de keras
train_ds = tf.keras.utils.image_dataset_from_directory(
    flowers_path,
    color_mode='rgb',
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    flowers_path,
    color_mode='rgb',
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size)

# Autotune [Optimización automática de recursos]
train = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("max_trial: {}".format(hyperband_max_epochs * (math.log(hyperband_max_epochs, factor) ** 2)))

# Hyperparam - Tuner
hypermodel = HyperModel(
    input_shape=input_shape, 
    output_size=len(labels))
tuner = kt.Hyperband(
    hypermodel,
    objective="val_accuracy",
    max_epochs = hyperband_max_epochs,
    factor=factor,
    seed=seed,
    hyperband_iterations=1,
    directory="Hyperband",
    project_name="flowers"
)

# Resumen
print(tuner.search_space_summary())

# Busqueda
tuner.search(train, epochs=hyperband_max_epochs, validation_data=val)

# Modelo CNN
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
earlystopping = tf.keras.callbacks.EarlyStopping(
      monitor="val_loss",
      min_delta=0,
      patience=20,
      verbose=0,
      mode="auto",
      baseline=None,
      restore_best_weights=True,
    )

model.fit(train, epochs=epochs, validation_data=val, callbacks=[earlystopping])
# Modelo final (Añadimos MLP)
model = Sequential(model.layers[:-1])
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dense(len(labels)))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
history = model.fit(train, epochs=epochs, validation_data=val, callbacks=[earlystopping])
print(model.summary())
plot_model(model, to_file='model.png')

# Visualización
loss_values = history.history['loss']
loss_val_values = history.history['val_loss']
acc_values = history.history['accuracy']
acc_val_values = history.history['val_accuracy']
epochs = range(len(loss_values))

mpl.style.use('seaborn')
fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(12, 6), constrained_layout=True)
axs[0].plot(epochs, loss_values, label='Training Loss')
axs[0].plot(epochs, loss_val_values, label='Validation Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].set_title("Loss history")
axs[0].legend()
axs[1].plot(epochs, acc_values, label='Training Acc')
axs[1].plot(epochs, acc_val_values, label='Validation Acc')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Acc')
axs[1].set_title("Acc history")
axs[1].legend()
fig.suptitle('History')
fig.savefig('Result')

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

