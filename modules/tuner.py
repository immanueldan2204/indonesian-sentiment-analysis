
from typing import NamedTuple, Dict, Text, Any, List
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

from keras.layers import (
    Input,
    concatenate,
    Dense,
    Dropout
)
from keras.callbacks import EarlyStopping
import kerastuner as kt
from keras_tuner.engine import base_tuner

from modules.transform import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    LABEL_KEY,
    transformed_name
)

# Define the result type for the tuner function.
TunerFnResult = NamedTuple('TunerFnResult', [
    ('tuner', base_tuner.BaseTuner), 
    ('fit_kwargs', Dict[Text, Any])
])

# Configure early stopping to prevent overfitting.
early_stopping = EarlyStopping(
    monitor='val_binary_accuracy', 
    patience=3,
    verbose=1, 
    mode='max'
)

def gzip_reader_fn(filenames):
    '''Loads compressed data.'''
    
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(
        file_pattern,
        tf_transform_output,
        batch_size=128
    ) -> tf.data.Dataset:
    '''
    Generates features and labels for tuning/training.
    
    Args:
        file_pattern (str): Input tfrecord file pattern.
        tf_transform_output: A TFTransformOutput.
        batch_size (int): Number of consecutive elements of returned dataset.

    Returns:
        A dataset that provides (features, label) tuples.
    '''

    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY)
    )

    return dataset

def tuner_model(hp):
    '''
    Builds the model for hyperparameter tuning.

    Args:
        hp: Hyperparameters object.

    Returns:
        tf.keras.Model: Compiled model.
    '''

    # Define input layers.
    input_features = []

    for feature in CATEGORICAL_FEATURES + NUMERICAL_FEATURES:
        input_features.append(
            Input(shape=(1,), name=transformed_name(feature))
        )

    # Concatenate inputs and pass through Sequential layers.
    concatenated = concatenate(input_features)

    # Define dense layers as a Sequential model.
    dense_layers = tf.keras.models.Sequential([
        Dense(
            hp.Choice('dense_units_1', values=[128, 256, 512]), 
            activation='relu'
        ),
        Dropout(hp.Float('dropout_rate_1', min_value=0.1, max_value=0.5, step=0.1)),
        Dense(
            hp.Choice('dense_units_2', values=[128, 256, 512]), 
            activation='relu'
        ),
        Dropout(hp.Float('dropout_rate_2', min_value=0.1, max_value=0.5, step=0.1)),
        Dense(1, activation='sigmoid')
    ], name='dense_stack')
    
    outputs = dense_layers(concatenated)

    # Build and compile model.
    model = tf.keras.Model(inputs=input_features, outputs=outputs)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    return model

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    '''
    Tune the hyperparameters for the model.

    Args:
        fn_args: Holds args as name/value pairs.
    
    Returns:
        A namedtuple contains the tuner and fit_kwargs.
    '''

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Prepare training and validation datasets.
    train_set = input_fn(fn_args.train_files[0], tf_transform_output)
    val_set = input_fn(fn_args.eval_files[0], tf_transform_output)

    # Initialize the tuner.
    tuner = kt.Hyperband(
        tuner_model,
        objective='val_binary_accuracy',
        max_epochs=5,
        factor=3,
        directory=fn_args.working_dir,
        project_name='diabetes_kt_hyperband'
    )

    # Return the tuner and fit arguments.
    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_set,
            'validation_data': val_set,
            'steps_per_epoch': 1000,
            'validation_steps': 1000,
            'callbacks': [early_stopping]
        }
    )
