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
            imply all preprocess function on the train data
            
        4. drop_dup_and_columns: 
            drop duplicate sample and non relevant columns.
            
        5.keyword_preprocess: 
            preprocess relevant to  keyword column
            
        6.text_train_preprocess
            preprocess relevant to text column on the train data
        
        7.text_test_preprocess
            preprocess relevant to text column on the test data
            
        7. clean text 
            clean the text from symbols space and etc
        
        8.preprocess_test: 
            imply all preprocess function on the test data
            
    Globals: 
    1. tokenizer
    2. stop_words_set
    3. model
    
    drop columns: 
    1. id 
    2. location 
    
    TODO: 
    1.check if it good to delete the location columns
    2.complete test preprocess
    
"""

######## imports ########
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
from nltk.corpus import stopwords
import re
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, LSTM

########## globals ##########
tokenizer = Tokenizer()
stop_words_set = set(stopwords.words('english'))
model = Sequential()


##### preprocess code ######

def split_data(data: str):
    """
    Shuffles the data and splits it into 90% train data and 10% validation data.

    :param data: str
        Path or file name of the CSV file containing the data.
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
     Converts a CSV file to a DataFrame.

     Parameters:
     :param csv_data: str
         Path or file name of the CSV file to be read.

     Returns:
     :return: pandas.DataFrame
         DataFrame containing the data from the CSV file.
     """
    data = pd.read_csv(csv_data)
    return data


def drop_dup_and_columns(data: pd.DataFrame):
    """
    Drops non-relevant columns and duplicate rows from the given DataFrame.

    Parameters:
    :param data: pd.DataFrame
        DataFrame containing the data to be processed.

    Returns:
    :return: pd.DataFrame
        DataFrame with duplicate rows removed and non-relevant columns ('id' and 'location') dropped.
    """
    data = data.drop_duplicates()
    data = data.drop(columns=['id', 'location'])  # drop id and location columns
    return data


def keyword_preprocess(data: pd.DataFrame):
    """
    preprocess on the keyword,
    applying get dummies for this column,
    (because for similar sample text there is the same keyword,and probably will get the same label)

    :param data: pandas.DataFrame
        DataFrame containing the 'keyword' column to be preprocessed.

    :return: pandas.DataFrame
        DataFrame with the 'keyword' column preprocessed using get dummies.
    """
    data = pd.get_dummies(data, columns=["keyword"])
    return data


def text_train_preprocess(data: pd.DataFrame):
    """
        Preprocesses the text data in the 'text' column of the given DataFrame.

        :param data: pandas.DataFrame
            DataFrame containing the text data to be preprocessed. It should have a 'text' column
            containing the preprocessed and tokenized text data.

        :return: pandas.DataFrame
            DataFrame with word embeddings as additional columns along with the other original columns
            from the input DataFrame.

        This function takes a DataFrame 'data' containing a column of preprocessed text data and performs
        the following tasks:
        1. Cleans the text data in the 'text' column using an 'clean_text' function
        2. Converts the tokenized and preprocessed text data to numerical form using the word-to-index mapping.
        3. Applies padding or truncating to sequences to ensure a fixed sequence length.
        4. Generates word embeddings using a pre-trained model based on the numerical sequences.

    """

    data = clean_text(data)
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
    model.add(Embedding(len(word_to_index) + 1, embedding_dim, input_length=max_sequence_length))
    model.add(LSTM(64))

    # Get word embeddings from the model
    word_embeddings = model.predict(padded_sequences)
    # Convert word embeddings to DataFrame
    column_titles = [f'embedding_{i}' for i in range(64)]
    word_embeddings_df = pd.DataFrame(word_embeddings, columns=column_titles)
    data = pd.concat([word_embeddings_df, data.drop(columns=['text'])], axis=1)
    return data


def text_test_preprocess(test_data):
    """
      Preprocesses the text data in the 'text' column of the given DataFrame.

      :param test_data: pd.DataFrame
          DataFrame containing the text data to be preprocessed. It should have a 'text' column
          containing the preprocessed and tokenized text data.

      :return: pd.DataFrame
          DataFrame with word embeddings as additional columns along with the other original columns
          from the input DataFrame.

      This function takes a DataFrame 'test_data' containing a column of preprocessed text data and performs
      the following tasks:
      1. Cleans the text data in the 'text' column using an 'clean_text' function
      2. Converts the tokenized and preprocessed text data to numerical form using the word-to-index mapping
         of the train data.
      3. Applies padding or truncating to sequences to ensure a fixed sequence length.
      4. Generates word embeddings using a pre-trained model based on the numerical sequences.

    """
    test_data = clean_text(test_data)
    # convert the text to numrical by the mapping of the train set
    convertNumerical = tokenizer.texts_to_sequences(test_data['text'])
    max_sequence_length = 50
    padded_sequences = pad_sequences(convertNumerical, maxlen=max_sequence_length, padding='post', truncating='post')
    # predict by the train set model
    word_embeddings = model.predict(padded_sequences)
    column_titles = [f'embedding_{i}' for i in range(64)]
    word_embeddings_df = pd.DataFrame(word_embeddings, columns=column_titles)
    test_data = pd.concat([word_embeddings_df, test_data.drop(columns=['text'])], axis=1)
    return test_data


def clean_text(data: pd.DataFrame):
    """
       Perform text preprocessing on the 'text' column of the given DataFrame.

       Inner Functions:
       1. lower_text(tweet: str) -> str: Convert the tweet to lowercase.
       2. clean_symbols(tweet: str) -> str: Remove symbols, URLs, mentions, hashtags, RT tags, HTML entities, and emojis
          from the tweet.
       3. split_text(tweet: str) -> List[str]: Split the tweet into individual words.
       4. clean_stop_words(split_tweet: List[str]) -> List[str]: Remove stopwords from the list of words.

       Parameters:
       :param data: pandas.DataFrame
           DataFrame containing the text data to be preprocessed. It should have a 'text' column
           containing the preprocessed text data as strings.

       Returns:
       :return: pandas.DataFrame
           DataFrame with preprocessed text in the 'text' column after removing symbols, URLs, mentions, hashtags,
            RT tags, HTML entities, emojis, and stopwords.
    """

    def lower_text(tweet):
        """
        lower all the letters in the tweet
        :param tweet: str
            Current tweet to work on.
        :return: str
            Tweet with lowercase letters.
        """
        return tweet.lower()

    def clean_symbols(tweet):
        """
        remove all the symbols from the tweet
        :param tweet: str
            Current tweet to work on.
        :return: str
            Tweet after removing symbols and other entities.
        """
        # Remove URLs
        tweet = re.sub(r'http\S+', '', tweet)

        # Remove Twitter mentions (@username)
        tweet = re.sub(r'@\w+', '', tweet)

        # Remove hashtags maintain the word (#word)
        tweet = re.sub(r'#(\w+)', r'\1', tweet)

        # Remove RT (Retweet) tag
        tweet = re.sub(r'\bRT\b', '', tweet)

        # Remove HTML entities like &gt;
        tweet = re.sub(r'&\w+;', '', tweet)

        # Remove emojis
        tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)

        # Remove non-letter characters, excluding whitespace
        tweet = re.sub(r'[^\w\s]', '', tweet)

        # Remove extra spaces and trim leading/trailing spaces
        tweet = re.sub(r'\s+', ' ', tweet).strip()
        return tweet

    def split_text(tweet):
        """
        Split the tweet into individual words.
        :param tweet: str
            Current tweet to work on.
        :return: List[str]
            List of words extracted from the tweet.
        """
        # split the text pre space
        return tweet.split()

    def clean_stop_words(split_tweet):
        """
        Remove stopwords from the list of words.
        :param split_tweet: List[str]
            List of words in the tweet.
        :return: List[str]
            List of words after removing stopwords.
        """
        # clean the stop words and conjunctions.
        split_tweet = [word for word in split_tweet if word not in stop_words_set]
        return split_tweet

    # make the text lower case
    data['text'] = data['text'].apply(lower_text)

    # clean symbols
    data['text'] = data['text'].apply(clean_symbols)

    # split the text
    data['text'] = data['text'].apply(split_text)

    # clean stop words
    data['text'] = data['text'].apply(clean_stop_words)

    # combine words back to sentence
    data['text'] = data['text'].apply(lambda words: ' '.join(words))

    return data


def train_preprocess(train_data: pd.DataFrame, y_train: pd.Series):
    """
     Preprocesses the given training data and target labels.

     :param train_data: pandas.DataFrame
         DataFrame containing the training data, where each row represents a sample and each column represents a feature.

     :param y_train: pandas.DataFrame
         DataFrame containing the target labels corresponding to the training data.

     :return: Tuple[pd.DataFrame, pd.Series, pd.Series]
         A tuple containing three elements:
         1. train_X: pd.DataFrame
             DataFrame containing the preprocessed features (excluding the target labels).
         2. train_y: pd.Series
             DataFrame containing the target labels.
         3. avg: pd.Series
             Series containing the average values of each feature in 'train_X' after preprocessing.
    """

    # concat the result column to the data
    data = pd.concat([train_data, y_train], axis=1)

    # drop the columns and drop duplicate samples
    data = drop_dup_and_columns(data)

    # preprocess on the keyword column
    data = keyword_preprocess(data)

    # preprocess on the text column
    data = text_train_preprocess(data)
    train_y = data.loc[:, y_train.name]
    train_X = data.drop(columns=y_train.name)
    avg = train_X.mean(axis=0)
    return train_X, train_y, avg


def preprocess_test(test_data: pd.DataFrame, avg_train):
    """
       Preprocesses the given test data to prepare it for prediction using a trained model.

       :param test_data: pd.DataFrame
           DataFrame containing the test data, where each row represents a sample and each column represents a feature.

       :param avg_train: pd.Series
           Series containing the average values of each feature in the training data after preprocessing.
            The index of the Series should match the column names of the preprocessed training data.

       :return: pd.DataFrame
           DataFrame containing the preprocessed test data, aligned with the features used for training.
    """

    # drop id and location columns
    test_data = test_data.drop(columns=['id', 'location'])

    # preprocess on the keyword column
    test_data = keyword_preprocess(test_data)

    # preprocess on the text column
    test_data = text_test_preprocess(test_data)

    # match the column of the test to the column of the train
    test_data = test_data.reindex(columns=avg_train.index, fill_value=0)
    return test_data