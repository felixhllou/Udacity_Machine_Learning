import numpy as np
import gensim
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras import regularizers
from keras.layers import Activation, Input, Dense, Flatten, Dropout, Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate

def create_tokenizer(reviews):
    """
    Fit a tokenizer.
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(reviews)
    
    return tokenizer

def max_length(reviews):
    """
    Calculate the maximum review length.
    """
    return max([len(review) for review in reviews])
 
def encode_reviews(tokenizer, reviews, length):
    """
    Encode a list of reviews.
    """
    # integer encode
    encoded = tokenizer.texts_to_sequences(reviews)
    # pad encoded sequences to the maximum length of review 
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    
    return padded

def create_channel(embedding, filter_size, feature_map):
    """
    Create a cnn channel
    """
    conv = Conv1D(feature_map, kernel_size=filter_size, activation='relu',
        strides=1, padding='same', kernel_regularizer=regularizers.l2(0.03))(embedding)
    pool = MaxPooling1D(pool_size=2, strides=1, padding='valid')(conv)
    flat = Flatten()(pool)
    
    return flat

def create_embed_layer(word_dict, num_of_words, embedding_dim, embed_model):
    """
    Create an embedding layer based on pretrained model.
    """
    # initialize embedding matrix
    embedding_matrix = np.zeros((num_of_words, embedding_dim))
    word_dict = word_dict # word dictionary
    # load pretrained embedding model
    word_vectors = KeyedVectors.load(embed_model)
    
    for word, idx in word_dict.items():
        try:
            embedding_vector = word_vectors[word]
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector
        except KeyError: # in case vocab size do not match
            embedding_matrix[idx] = np.random.normal(0, np.sqrt(0.25), embedding_dim)
    # create an embedding layer
    embedding_layer = Embedding(input_dim=num_of_words, output_dim=embedding_dim,
                            weights=[embedding_matrix], trainable=True
                           )
    
    return embedding_layer

def build_cnn(embed_model=None, num_of_words=None, word_dict=None,
              embedding_dim=None, filter_sizes=[3,4,5],
              feature_maps=[100,100,100], max_seq_length=100,
              activation='sigmoid', output_units=1,
              dropout_rate=None, loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'], model_image='multichannel.png'):
    """
    A CNN for text classification
    
    Arguments:
        embed_model     : A pretrained model or None
        num_of_words    : The size of the vocabulary
        word_dict       : A dictionary in the form of {vocab: idx}
        embedding_dim   : The dimension of word representations
        filter_sizes    : An array of ngrams per channel
        feature_maps    : An array of feature maps (patterns) per channel
        max_seq_length  : The maximum length of sequence (review)
        activation      : The activation function for output
        output_units    : The number of output units
        dropout_rate    : If defined, dropout will be added after embedding layer & concatenation
        loss            : Loss function
        optimizer       : Optimizer
        metrics         : Evaluation metrics
        model_image     : Model image output saved path
        
    Returns:
        Model           : Keras model instance
    """
    
    # checkup
    if embed_model and word_dict is None:
        raise Exception('Define `word_dict {vocab: idx}` that matches the `num_of_words`.')
    if len(filter_sizes)!=len(feature_maps):
        raise Exception('`filter_sizes` and `feature_maps` must have the same length.')
    if not embed_model and (not num_of_words or not embedding_dim):
        raise Exception('Define `num_of_words` and `embedding_dim` if there is no pretrained embedding')
    
    print('Creating CNN ...')
    print('****************')
    print('Embedding: %s pretrained embedding' % ('using' if embed_model else 'no'))
    print('Vocabulary size: %s' % num_of_words)
    print('Embedding dim: %s' % embedding_dim)
    print('Filter sizes: %s' % filter_sizes)
    print('Feature maps: %s' % feature_maps)
    print('Max sequence: %i' % max_seq_length)
    print('****************')

    # create an embedding layer
    if embed_model is not None:
        embedding_layer = create_embed_layer(word_dict, num_of_words, embedding_dim, embed_model)
    else:
        embedding_layer = Embedding(input_dim=num_of_words, output_dim=embedding_dim,
                                    input_length=max_seq_length,
                                    weights=None,
                                    trainable=True
                                   )
    
    # initialize a list for channel(s)
    channels = []

    # inputs and embedding layer
    inputs = Input(shape=(max_seq_length,), dtype='int32')
    embedding = embedding_layer(inputs)
    
    # check if there is any dropout rate
    if dropout_rate:
       embedding = Dropout(dropout_rate)(embedding)
    
    # create cnn channel(S)
    for idx, size in enumerate(filter_sizes):
        flat = create_channel(embedding=embedding, filter_size=size, feature_map=feature_maps[idx])
        channels.append(flat)
    
    # concatenate all channel(s)
    merged = concatenate(channels)
    if dropout_rate:
        merged = Dropout(dropout_rate)(merged)
    
    # final interpretation
    dense = Activation('relu')(merged)
    outputs = Dense(units=output_units, activation=activation)(dense)
    model = Model(inputs=[inputs], outputs=outputs)
    
    # compile model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # summarize
    print(model.summary())
    plot_model(model, show_shapes=True, to_file=model_image)
    
    return model
