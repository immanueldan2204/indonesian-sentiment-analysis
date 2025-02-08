
# Import required libraries.
import tensorflow as tf

# Define feature and label keys.
LABEL_KEY = 'label'
FEATURE_KEY = 'text'

def transformed_name(key):
    '''
    Rename transformed features.

    Args:
        key (str): Feature name to be transformed.
    
    Returns:
        str: Transformed feature name.
    '''

    return key + '_xf'

def preprocessing_fn(inputs):
    '''
    Preprocess input features into transformed features.
    
    Args:
        inputs (dict): Map from feature keys to raw features.
        
    Returns:
        outputs (dict): Map from feature keys to transformed features.
    '''
    outputs = {}
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    
    return outputs
