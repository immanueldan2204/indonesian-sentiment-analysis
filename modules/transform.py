'''Transform module.'''

# Import required libraries.
import tensorflow as tf
import nltk
from nltk.corpus import stopwords

# Download stopwords and load them for preprocessing.
nltk.download('stopwords')
stopwords = list(stopwords.words('english'))

# Define feature and label keys.
LABEL_KEY = 'generated'
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

    # Standardize and clean text data.
    text = tf.strings.lower(inputs[FEATURE_KEY])
    text = tf.strings.regex_replace(text, r'[^a-z\s]', ' ')
    text = tf.strings.regex_replace(text, r'\b(' + r'|'.join(stopwords) + r')\b\s*', ' ')
    text = tf.strings.strip(text)
    
    # Assign transformed features to outputs.
    outputs[transformed_name(FEATURE_KEY)] = text
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    
    return outputs
