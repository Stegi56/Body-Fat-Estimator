import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

print("Imported modules.")

dataFrame = pd.read_csv("bodyfat.csv")
dataFrame = dataFrame.reindex(np.random.permutation(dataFrame.index))

trainDF = dataFrame.sample(frac = 0.8)
testDF = dataFrame.drop(trainDF.index)

trainDF

# Keras Input tensors of float values.
inputs = {
    'density':
        tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                              name='density'),
    'bodyfat':
        tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                              name='bodyfat'),
    'age':
        tf.keras.layers.Input(shape=(1,), dtype=tf.int8,
                              name='age'),
    'weight':
        tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                              name='weight'),
    'height':
        tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                              name='height'),
    'neck':
        tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                              name='neck'),
    'chest':
        tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                              name='chest'),
    'abdomen':
        tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                              name='abdomen'),
    'hip':
        tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                              name='hip'),
    'thigh':
        tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                              name='thigh'),
    'knee':
        tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                              name='knee'),
    'ankle':
        tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                              name='ankle'),
    'biceps':
        tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                              name='biceps'),
    'forearm':
        tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                              name='forearm'),
    'wrist':
        tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                              name='wrist')
}

#Normalise
density = tf.keras.layers.Normalization(
    name = 'normalization_density',
    axis=None)
density.adapt(trainDF['Density'])
density = density(inputs.get('density'))

bodyfat = tf.keras.layers.Normalization(
    name = 'normalization_bodyfat',
    axis=None)
density.adapt(trainDF['BodyFat'])
density = density(inputs.get('bodyfat'))

age = tf.keras.layers.Normalization(
    name = 'normalization_age',
    axis=None)
density.adapt(trainDF['Age'])
density = density(inputs.get('age'))

weight = tf.keras.layers.Normalization(
    name = 'normalization_weight',
    axis=None)
density.adapt(trainDF['Weight'])
density = density(inputs.get('weight'))

height = tf.keras.layers.Normalization(
    name = 'normalization_height',
    axis=None)
density.adapt(trainDF['Height'])
density = density(inputs.get('height'))

neck = tf.keras.layers.Normalization(
    name = 'normalization_neck',
    axis=None)
density.adapt(trainDF['Neck'])
density = density(inputs.get('neck'))

chest = tf.keras.layers.Normalization(
    name = 'normalization_chest',
    axis=None)
density.adapt(trainDF['Chest'])
density = density(inputs.get('chest'))

abdomen = tf.keras.layers.Normalization(
    name = 'normalization_abdomen',
    axis=None)
density.adapt(trainDF['Abdomen'])
density = density(inputs.get('abdomen'))

hip = tf.keras.layers.Normalization(
    name = 'normalization_hip',
    axis=None)
density.adapt(trainDF['Hip'])
density = density(inputs.get('hip'))

thigh = tf.keras.layers.Normalization(
    name = 'normalization_thigh',
    axis=None)
density.adapt(trainDF['Thigh'])
density = density(inputs.get('thigh'))

knee = tf.keras.layers.Normalization(
    name = 'normalization_knee',
    axis=None)
density.adapt(trainDF['Knee'])
density = density(inputs.get('knee'))

ankle = tf.keras.layers.Normalization(
    name = 'normalization_ankle',
    axis=None)
density.adapt(trainDF['Ankle'])
density = density(inputs.get('ankle'))

biceps = tf.keras.layers.Normalization(
    name = 'normalization_biceps',
    axis=None)
density.adapt(trainDF['Biceps'])
density = density(inputs.get('biceps'))

forearm = tf.keras.layers.Normalization(
    name = 'normalization_forearm',
    axis=None)
density.adapt(trainDF['Forearm'])
density = density(inputs.get('forearm'))

wrist = tf.keras.layers.Normalization(
    name = 'normalization_wrist',
    axis=None)
density.adapt(trainDF['Wrist'])
density = density(inputs.get('wrist'))

# Concatenate our inputs into a single tensor.
preprocessing_layers = tf.keras.layers.Concatenate()()

print("Preprocessing layers defined.")