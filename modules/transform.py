
import tensorflow as tf
import tensorflow_transform as tft

CATEGORICAL_FEATURES = [
    'high_bp',
    'high_chol',
    'chol_check',
    'smoker',
    'stroke',
    'heart_disease_or_attack',
    'phys_activity',
    'fruits',
    'veggies',
    'hvy_alcohol_consump',
    'any_healthcare',
    'no_docbc_cost',
    'diff_walk',
    'sex'
]

NUMERICAL_FEATURES = [
    'bmi',
    'gen_hlth',
    'ment_hlth',
    'phys_hlth',
    'age',
    'education',
    'income'
]

LABEL_KEY = 'diabetes'

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

    for feature in CATEGORICAL_FEATURES:
        outputs[transformed_name(feature)] = tf.cast(inputs[feature], tf.int64)

    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
    
