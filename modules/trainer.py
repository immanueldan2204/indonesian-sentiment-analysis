
import os

import tensorflow as tf
import tensorflow_transform as tft
from keras.utils.vis_utils import plot_model
from keras.layers import (
    Input,
    concatenate,
    Dense,
    BatchNormalization,
    Dropout
)
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

from modules.transform import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    LABEL_KEY,
    transformed_name
)
from modules.tuner import gzip_reader_fn, input_fn

def trainer_model(hp):
    '''
    Builds the model for training.

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
        Dense(hp['dense_units'], activation='relu'),
        BatchNormalization(),
        Dropout(hp['dropout_rate']),
        Dense(1, activation='sigmoid')
    ], name='dense_stack')
    
    outputs = dense_layers(concatenated)

    # Build and compile model.
    model = tf.keras.Model(inputs=input_features, outputs=outputs)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(hp['learning_rate']),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    model.summary()

    return model

def get_serve_tf_examples_fn(model, tf_transform_output):
    '''Returns a function that parses a serialized tf.Example.'''

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        '''Returns the output to be used in the serving signature.'''

        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec
        )
        transformed_features = model.tft_layer(parsed_features)
        outputs = model(transformed_features)

        return {'outputs': outputs}
    
    return serve_tf_examples_fn

def run_fn(fn_args) -> None:
    '''
    Train the model based on given args.
    
    Args:
        fn_args: Holds args used to train the model as name/value pairs.
    '''

    # Load the transform output.
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    # Prepare training and validation datasets.
    train_set = input_fn(fn_args.train_files, tf_transform_output)
    val_set = input_fn(fn_args.eval_files, tf_transform_output)

    # Build model.
    hp = fn_args.hyperparameters['values']
    model = trainer_model(hp)

    # Define callbacks.
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    tensorboard = TensorBoard(log_dir=log_dir, update_freq='batch')
    early_stopping = EarlyStopping(
        monitor='val_binary_accuracy', 
        patience=3, 
        verbose=1, 
        mode='max', 
        restore_best_weights=True
    )
    model_checkpoint = ModelCheckpoint(
        fn_args.serving_model_dir,
        monitor='val_binary_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    # Train model.
    model.fit(
        train_set,
        steps_per_epoch=fn_args.train_steps,
        validation_data=val_set,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard, early_stopping, model_checkpoint],
        epochs=10
    )

    # Define and save model signatures for serving
    signatures = {
        'serving_default': get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
        )
    }

    model.save(
        fn_args.serving_model_dir, save_format='tf', signatures=signatures
    )

    # Save model plot.
    plot_model(
        model, 
        to_file='images/model_plot.png', 
        show_shapes=True, 
        show_layer_names=True
    )
