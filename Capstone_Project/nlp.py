import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from gensim.parsing.preprocessing import strip_punctuation, strip_non_alphanum, remove_stopwords, strip_short

wc_stopwords = set(STOPWORDS)
size = (20, 10)

def clean_reviews(reviews, concat=False):
    """
    clean and tokenize reviews (Concatenate if necessary).
    """
    # remove stopwords from reviews
    reviews = [remove_stopwords(review) for review in reviews]
    # remove words that are not alphabetic
    reviews = [strip_non_alphanum(review) for review in reviews]
    # remove punctuation from reviews
    reviews = [strip_punctuation(review) for review in reviews]
    # filter out short words
    reviews = [strip_short(review, minsize=2) for review in reviews]
    # stopwords
    stop_words = set(stopwords.words('english'))
    # concat all reviews if `True`
    if concat is True:
        reviews = ' '.join(review.lower() for review in reviews)
        tokenized_words = word_tokenize(reviews)
        tokenized_words = [word for word in tokenized_words if not word in stop_words]
    else:
        # tokenize each review into individual words
        tokenized_words = [word_tokenize(review.lower()) for review in reviews]
        tokenized_words = [[word for word in review if not word in stop_words] for review in tokenized_words]
    
    return tokenized_words

def word_count(reviews, num_of_words):
    """
    Construct a word count Dataframe.
    
    Reference:
    https://www.kaggle.com/nicapotato/guided-numeric-and-text-exploration-e-commerce
    """
    word_count = nltk.FreqDist(reviews)
    top_N = num_of_words
    rslt = pd.DataFrame(word_count.most_common(top_N), 
                        columns=['Word', 'Frequency']).set_index('Word')
    
    return rslt

def wc_plot(reviews, title, stopwords=stopwords, size=size):
    """
    Function to plot WordCloud.

    Reference:
    https://www.kaggle.com/nicapotato/guided-numeric-and-text-exploration-e-commerce
    https://www.kaggle.com/longdoan/word-cloud-with-python?scriptVersionId=728708
    """
    # figure parameters
    mpl.rcParams['figure.figsize']=(10.0,10.0)
    mpl.rcParams['font.size']=12
    mpl.rcParams['savefig.dpi']=100
    mpl.rcParams['figure.subplot.bottom']=.1 
    
    # instantiate WordCloud object
    wordcloud = WordCloud(width=1600, height=800,
                          background_color='black',
                          stopwords=stopwords).generate(str(reviews))
    
    # visual's parameters
    fig = plt.figure(figsize=size, facecolor='k')
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(title, fontsize=50,color='y')
    plt.tight_layout(pad=0)
    plt.show()

def gen_ngrams(reviews, degree):
    """
    Return the ngrams generated from a sequence of items
    and join each gram.

    Reference:
    https://www.kaggle.com/nicapotato/guided-numeric-and-text-exploration-e-commerce
    """
    n_grams = ngrams((reviews), degree)
    
    return [' '.join(grams) for grams in n_grams]

def gram_count(reviews, degree, top_n):
    """
    Return a Dataframe on ngrams count.

    Reference:
    https://www.kaggle.com/nicapotato/guided-numeric-and-text-exploration-e-commerce
    """
    # perform ngrams generation and join each of them (Each bite of text)
    joined_grams = gen_ngrams(reviews, degree)
    
    # counter object on measuring each gram's occurrence
    grams_count = Counter(joined_grams)
    
    # tranform the counter object to a Dataframe
    df = pd.DataFrame.from_dict(grams_count, orient='index')
    df = df.rename(columns={'index': 'words', 0: 'frequency'}) # Renaming index column name
    
    return df.sort_values(["frequency"], ascending=[0])[:top_n]

def gram_lookup(reviews, degree, top_n):
    """
    Return a lookup table on gram count.

    Reference:
    https://www.kaggle.com/nicapotato/guided-numeric-and-text-exploration-e-commerce
    """
    # initialize a lookup dataframe
    lookup = pd.DataFrame(index=None)
    
    # loop over degrees of ngrams
    for d in degree:
        table = pd.DataFrame(gram_count(clean_reviews(reviews, concat=True), d, top_n).reset_index()) # ngrams count
        table.columns = ['{}-Gram'.format(d), 'Count'] # rename columns
        lookup = pd.concat([lookup, table], axis=1) # concatenate gram tables
    
    return lookup