"""    ##########  file description  ###########  """
"""
    target : Preprocessing for a disaster prediction project from tweets involves text cleaning,
    tokenization, and vectorization to convert raw text data into a suitable format
    for machine learning.
    The process ensures relevant features are extracted and noise is eliminated to enhance the model's accuracy. 
    
    
    function: 
        1. split_data :
            shuffle the data and split it to validation and train data (we have already test data)
            
        2. convert_to_csv:
            convert a csv file to DataFrame
            
        3. train_preprocess: 
            imply all th preprocess function on the train data
            
        4. drop_dup_and_columns: 
            drop duplicate sample and non relevant columns.
            
        5.keyword_preprocess: 
            preprocess relevant to  keyword column
            
        6.text_preprocess
            preprocess relevant to text column
            
        7. clean text 
            clean the text from symbols space and etc
            
    Globals: 
    1. tokenizer
    
    drop columns: 
    1. id 
    2. location 
    
    TODO: 
    1. check if it good to delete the location columns
    
"""
######## imports ########
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
from keras.utils import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, LSTM

#########################

########## globals ##########
tokenizer = Tokenizer()
#############################

def split_data(data: str):
    """
    the function shuffle the data and split it to:
    90% train data
    10% validation data.
    :param data: the data which we work on
    """
    data = convert_to_csv(data)
    data = data.sample(frac=1)  # shuffle the data
    split_point = int(0.9 * len(data))
    train_data = data[:split_point]
    validation_data = data[split_point:]
    train_data.to_csv("S_train_data.csv", index=False)
    validation_data.to_csv("validation_data.csv", index=False)


def convert_to_csv(csv_data: str):
    """
    convart csv file to DataFrame
    :param csv_data:
    :return: DataFrame
    """
    data = pd.read_csv(csv_data)
    return data


def drop_dup_and_columns(data: pd.DataFrame):
    """
    the function drop the non relevent columns and drop duplicate rows
    :param data: current working data
    :return: current data after changes
    """
    data = data.drop_duplicates()
    data = data.drop(columns=['id', 'location'])  # drop id and location columns
    return data


def keyword_preprocess(data: pd.DataFrame):
    """
    preprocess on the keyword,
    applying get dummies for this column, because there for similar sample text there is the same
    keyword, and probably will get the same label.
    :param data:
    :return:
    """
    data = pd.get_dummies(data, columns=["keyword"])
    return data


def text_preprocess(data: pd.DataFrame, stop_words_set: set):
    """
    Preprocesses the text data in the 'text' column of the given DataFrame.

    :param data: DataFrame containing the text data to be preprocessed.
    :param stop_words_set: A set of stopwords to be removed from the text data.

    :return: DataFrame with word embeddings as additional columns along with the other original columns.
    """
    data = clean_text(data, stop_words_set)
    # Step 1: Create Word-to-Index Mapping
    tokenizer.fit_on_texts(data['text'])
    word_to_index = tokenizer.word_index

    # Step 2: Convert Text Data to Numerical Form
    numerical_sequences = tokenizer.texts_to_sequences(data['text'])

    # Step 3: Padding or Truncating Sequences (Optional)
    max_sequence_length = 50
    padded_sequences = pad_sequences(numerical_sequences, maxlen=max_sequence_length, padding='post', truncating='post')

    # Step 4: Load Pre-trained Embeddings (Optional)
    embedding_dim = 100  # Adjust this according to the desired embedding size

    # Build the Model with Training from Scratch Word Embeddings
    model = Sequential()
    model.add(Embedding(len(word_to_index) + 1, embedding_dim, input_length=max_sequence_length))
    model.add(LSTM(64))

    # Get word embeddings from the model
    word_embeddings = model.predict(padded_sequences)
    # Convert word embeddings to DataFrame
    column_titles = [f'embedding_{i}' for i in range(64)]
    word_embeddings_df = pd.DataFrame(word_embeddings, columns=column_titles)
    data = pd.concat([word_embeddings_df, data.drop(columns=['text'])], axis=1)
    return data


def clean_text(data: pd.DataFrame, stop_words_set: set):
    """
    doing preprocess on the text column
    inner functions:
    1. lower_case
    2. clean_symbols
    3. split_text
    :param stop_words_set: A set of stopwords to be removed from the text data.
    :param data: current data
    :return: data after preprocess on the text columns
    """

    def lower_text(tweet):
        """
        lower all the letters in the tweet
        :param tweet: current tweet we work on
        :return: tweet with lowercase letters
        """
        return tweet.lower()

    def clean_symbols(tweet):
        """
        remove all the symbols from the tweet
        :param tweet: current tweet to work on
        :return: tweet after removing all the symbols.
        """
        # Remove URLs
        tweet = re.sub(r'http\S+', '', tweet)

        # Remove Twitter mentions (@username)
        tweet = re.sub(r'@\w+', '', tweet)

        # Remove hashtags (#word)
        tweet = re.sub(r'#(\w+)', r'\1', tweet)

        # Remove RT (Retweet) tag
        tweet = re.sub(r'\bRT\b', '', tweet)

        # Remove HTML entities like &gt;
        tweet = re.sub(r'&\w+;', '', tweet)

        # Remove emojis (You may need to adapt this regex to your specific case)
        tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)

        # Remove non-letter characters, excluding whitespace
        tweet = re.sub(r'[^\w\s]', '', tweet)

        # Remove extra spaces and trim leading/trailing spaces
        tweet = re.sub(r'\s+', ' ', tweet).strip()
        return tweet

    def split_text(tweet):
        return tweet.split()

    def clean_stop_words(split_tweet):
        split_tweet = [word for word in split_tweet if word not in stop_words_set]
        return split_tweet

    # make the text lower case.
    data['text'] = data['text'].apply(lower_text)

    # clean symbols
    data['text'] = data['text'].apply(clean_symbols)

    # split the text.
    data['text'] = data['text'].apply(split_text)

    # clean stop words
    data['text'] = data['text'].apply(clean_stop_words)

    # combine words back to sentence
    data['text'] = data['text'].apply(lambda words: ' '.join(words))

    #
    # tfidf_representation = vectorizer.fit_transform(data['text'])
    #
    # # Convert TF-IDF representation to DataFrame
    # tfidf_df = pd.DataFrame(tfidf_representation.toarray(), columns=vectorizer.get_feature_names_out())
    #
    # # Combine TF-IDF DataFrame with the other 256 columns
    # data = pd.concat([tfidf_df, data.drop(columns=['text'])], axis=1)

    # vectorize
    ## sec way - will create 2D dimension array per entry
    # vectorizer.fit_transform(data['text'])
    # data['text'] = data['text'].apply(lambda sentence: vectorizer.transform([sentence]).toarray())

    ### one way
    # bow_representation = vectorizer.fit_transform(data['text'])
    #
    # bow_df = pd.DataFrame(bow_representation.toarray(), columns=vectorizer.get_feature_names_out())
    #
    # # Combine BoW DataFrame with the other 256 columns
    # data = pd.concat([bow_df, data.drop(columns=['text'])], axis=1)
    ######

    return data


def train_preprocess(train_data: pd.DataFrame,y_train,stop_words: set):
    """
    :param stop_words:
    :param X_data: train data (without the label column)
    :param y_data: label column
    :return:
        1. train Data Frame after preprocess.
        2. label column after preprocess.
    """
    data = pd.concat([train_data, y_train], axis=1)  # concat the result column to the data
    data = drop_dup_and_columns(data)
    data = keyword_preprocess(data)
    data = text_preprocess(data, stop_words)
    train_y = data.loc[:, y_train.columns]
    train_X = data.drop(columns=y_train.columns)
    return train_X,train_y


if __name__ == '__main__':
    print("hello")