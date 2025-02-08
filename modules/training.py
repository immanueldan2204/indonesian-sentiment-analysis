'''Training module.'''

# Import TensorFlow, TFX, and utilities.
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
import os

# Import Keras components.
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
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils.vis_utils import plot_model

# Import constants and utilities from the transform and tuner modules.
from transform import FEATURE_KEY, LABEL_KEY, transformed_name
from sentiment_tuner import (
    VOCAB_SIZE, 
    SEQ_LENGTH, 
    EPOCHS,
    vectorize_layer, 
    gzip_reader_fn, 
    input_fn
)

# Define the model architecture with tunable hyperparameters.
def model_builder(hp):
    '''Build the machine learning model.'''
    
    model = Sequential([
        InputLayer(
            input_shape=(1,),
            dtype=tf.string, 
            name=transformed_name(FEATURE_KEY)
        ),
        Reshape(()), 
        vectorize_layer,
        Embedding(VOCAB_SIZE, hp['embedding_dim'], name='embedding'),
        SpatialDropout1D(hp['spatial_dropout_rate']),
        GlobalAveragePooling1D(),
        Dense(hp['dense_units'], activation='relu'),
        Dropout(hp['dropout_rate']),
        Dense(3, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(hp['learning_rate']),
        metrics=['accuracy']
    )

    model.summary()

    return model

# Define the serving function for the model.
def _get_serve_tf_examples_fn(model, tf_transform_output):
    '''Prepare the model for serving.'''
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_example):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_example, 
            feature_spec
        )
        transformed_features = model.tft_layer(parsed_features)

        # Generate predictions using transformed features.
        return model(transformed_features)

    return serve_tf_examples_fn

# Define the main function to train and save the model.
def run_fn(fn_args: FnArgs) -> None:
    '''Run the model training and evaluation.'''
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    
    # Configure callbacks for training.
    tensorboard = TensorBoard(log_dir=log_dir, update_freq='batch')
    early_stopping = EarlyStopping(
        monitor='val_accuracy', 
        patience=5,
        verbose=1,
        mode='max', 
        restore_best_weights=True
    )
    model_checkpoint = ModelCheckpoint(
        fn_args.serving_model_dir,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    # Load the transform output.
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Prepare training and validation datasets.
    train_set = input_fn(fn_args.train_files, tf_transform_output, EPOCHS)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, EPOCHS)

    # Adapt the TextVectorization layer using training data.
    vectorize_layer.adapt([
        j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)] for i in list(train_set)
        ]
    ])

    # Build the model using hyperparameters.
    hp = fn_args.hyperparameters['values']
    model = model_builder(hp)

    # Train the model with specified callbacks.
    model.fit(
        x=train_set,
        validation_data=val_set,
        callbacks=[tensorboard, early_stopping, model_checkpoint],
        epochs=EPOCHS
    )

    # Define and save model signatures for serving.
    signatures = {
        'serving_default': _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'
            )
        )
    }

    model.save(
        fn_args.serving_model_dir, 
        save_format='tf', 
        signatures=signatures
    )
