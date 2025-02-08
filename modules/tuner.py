'''Tuner module.'''

# Import necessary libraries and modules.
from typing import NamedTuple, Dict, Text, Any, List
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

# Import Keras and KerasTuner components.
from keras.models import Sequential
from keras.layers import (
    InputLayer,
    Reshape,
    TextVectorization,
    Embedding,
    SpatialDropout1D,
    GlobalAveragePooling1D,
    Dropout,
    Dense,
)
from keras.callbacks import EarlyStopping
import kerastuner as kt
from keras_tuner.engine import base_tuner

# Import constants and utility functions from the transform module.
from sentiment_transform import FEATURE_KEY, LABEL_KEY, transformed_name

# Define constants for vocabulary size and sequence length.
VOCAB_SIZE = 100000
SEQ_LENGTH = 1000
EPOCHS = 10

# Initialize a TextVectorization layer.
vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQ_LENGTH
)

# Define the result type for the tuner function.
TunerFnResult = NamedTuple('TunerFnResult', [
    ('tuner', base_tuner.BaseTuner), 
    ('fit_kwargs', Dict[Text, Any])
])

# Configure early stopping to prevent overfitting.
early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=3,
    verbose=1, 
    mode='max'
)

# Function to read compressed TFRecord files.
def gzip_reader_fn(filenames):
    '''Loads compressed data.'''
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

# Function to prepare input data for training and evaluation.
def input_fn(
        file_pattern,
        tf_transform_output,
        num_epochs,
        batch_size=128
    ) -> tf.data.Dataset:
    '''Get post_transform feature and create batches of data.'''

    # Get transformed feature specifications.
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    # Create batched dataset.
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY)
    )

    return dataset

# Function to define and build the model with tunable hyperparameters.
def model_builder(hp):
    '''Builds the model and sets up the hyperparameters to tune.'''
    
    # Define the model architecture.
    model = Sequential([
        InputLayer(
            input_shape=(1,),
            dtype=tf.string, 
            name=transformed_name(FEATURE_KEY)
        ),
        Reshape(()), 
        vectorize_layer,
        Embedding(
            VOCAB_SIZE, 
            hp.Int('embedding_dim', min_value=32, max_value=128, step=32),
            name='embedding'
        ),
        SpatialDropout1D(
            hp.Float('spatial_dropout_rate', min_value=0, max_value=0.5, step=0.1)
        ),
        GlobalAveragePooling1D(),
        Dense(
            hp.Int('dense_units', min_value=32, max_value=128, step=32),
            activation='relu'
        ),
        Dropout(hp.Float('dropout_rate', min_value=0, max_value=0.5, step=0.1)),
        Dense(3, activation='softmax')
    ])
    
    # Compile the model.
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice(
                'learning_rate', 
                values=[1e-2, 1e-3, 1e-4]
            )
        ),
        metrics=['accuracy']
    )

    return model
        
# Function to configure the tuner.
def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    '''Build the tuner using the KerasTuner API.'''
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    # Prepare training and validation datasets.
    train_set = input_fn(fn_args.train_files[0], tf_transform_output, EPOCHS)
    val_set = input_fn(fn_args.eval_files[0], tf_transform_output, EPOCHS)

    # Adapt the vectorization layer with training data.
    vectorize_layer.adapt([
        j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)] for i in list(train_set)
        ]
    ])

    # Initialize the tuner.
    tuner = kt.Hyperband(
        model_builder,
        objective='val_accuracy',
        max_epochs=EPOCHS,
        factor=3,
        directory=fn_args.working_dir,
        project_name='sentiment_analysis_kt_hyperband'
    )

    # Return the tuner and fit arguments.
    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'callbacks': [early_stopping],
            'x': train_set,
            'validation_data': val_set
        }
    )
