import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
"""
Every functions should take following two as inputs:

observation_space
action_space
"""

def eye_model(inputs, left_or_right):
    """
    Return an eye model
    Parameters
    ----------
    inputs : keras.Input

    left_or_right : str
    """
    x = layers.Reshape((inputs.shape[1],
                        inputs.shape[2]*inputs.shape[3]))(inputs)
    x = layers.Conv1D(64, 7, strides=1, activation='relu')(x)
    x = layers.Conv1D(128, 5, strides=2, activation='relu')(x)
    x = layers.Conv1D(192, 3, strides=2, activation='relu')(x)
    outputs = layers.Conv1D(256, 3, strides=2, activation='relu')(x)
    return keras.Model(inputs=inputs, outputs=outputs, 
                name=left_or_right+'_eye')

def brain_layers(action_n, x):
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(action_n)(x)
    return outputs

def mouse_eye_brain_model(observation_space, action_space):
    right_input = keras.Input(observation_space['Right'].shape,
                            name='Right')
    left_input = keras.Input(observation_space['Left'].shape,
                            name='Left')

    right_eye_model = eye_model(right_input, 'right')
    left_eye_model = eye_model(left_input, 'left')

    right_encoded = right_eye_model(right_input)
    left_encoded = left_eye_model(left_input)

    concat = layers.Concatenate()([left_encoded, right_encoded])
    outputs = brain_layers(action_space.n, concat)

    model = keras.Model(inputs=[left_input, right_input], outputs=outputs)

    return model

def cartpole_model(observation_space, action_space):
    inputs = keras.Input(observation_space['obs'].shape, name='obs')
    x = layers.Flatten()(inputs)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(action_space.n)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model