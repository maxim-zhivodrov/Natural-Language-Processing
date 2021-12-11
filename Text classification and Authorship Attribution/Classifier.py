import csv
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd
from datetime import datetime
import numpy as np
from nltk.corpus import stopwords
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from gensim import corpora

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from sklearn.metrics import classification_report

best_model_glob = None
best_vectorizer_glob = None
trained_model_glob = None
trained_vectorizer_glob = None


def main():
    """
    The main method was used for tests
    """

    # region Submission
    x = 0
    # model_file = open('model','wb')
    # global trained_vectorizer_glob
    # model_vectorizer_tpl = (train_best_model(), trained_vectorizer_glob)
    # pickle.dump(model_vectorizer_tpl, model_file)
    # model_file.close()
    #
    # best_model = load_best_model()
    #
    # predictions = predict(best_model, 'trump_test.tsv')
    # write_test_predictions(predictions)
    # endregion



def load_best_model():
    """
    Loads best model from pickle file
    Returns:
            (model): the model that was loaded
    """
    model_file = open('model', 'rb')
    model, vectorizer = pickle.load(model_file)
    model_file.close()
    global best_vectorizer_glob, best_model_glob
    best_vectorizer_glob = vectorizer
    best_model_glob = model
    return model

def train_best_model(fn='trump_train.tsv'):
    """
    Trains the best model of the four models
    Returns:
            (model): best model after training
    """
    tweets_df_train = read_tsv(fn , ['tweet_id', 'user_handle', 'tweet_text', 'time_stamp', 'device'])
    tweet_class_list = separte_tweets(tweets_df_train)  # Tuple: (pp tweet, class, org tweet)

    train_X_splitted = [tpl[0] for tpl in tweet_class_list]
    train_Y_splitted = [tpl[1] for tpl in tweet_class_list]
    original_train_tweets = [tpl[2] for tpl in tweet_class_list]

    vectorizer = CountVectorizer(stop_words='english', lowercase=True)
    train_X = form_features(vectorizer.fit_transform(train_X_splitted).toarray(), original_train_tweets)
    train_Y = train_Y_splitted

    LogReg_model = train_LogReg_model(train_X, train_Y)

    global trained_vectorizer_glob, trained_model_glob
    trained_vectorizer_glob = vectorizer
    trained_model_glob = LogReg_model
    return LogReg_model


def predict(m, fn):
    """
    Make predictions on the test set with the input model
    Args:
        m (model): trained model to predict with
        fn (str): path to test set
    Returns:
            (list): list of predictions
    """
    test_data = get_test_data(fn)
    test_X_splitted = [tpl[0] for tpl in test_data]
    original_test_tweets = [tpl[1] for tpl in test_data]

    vectorizer = None
    global trained_model_glob, trained_vectorizer_glob, best_model_glob, best_vectorizer_glob
    if m is trained_model_glob:
        vectorizer = trained_vectorizer_glob
    elif m is best_model_glob:
        vectorizer = best_vectorizer_glob

    test_X = form_features(vectorizer.transform(test_X_splitted).toarray(), original_test_tweets)
    return list(m.predict(test_X))

def train_LogReg_model(train_X, train_Y):
    """
    Trains Logistic regression model
    Args:
        train_X (list): list of tweets
        train_Y (list): list of classes
    Returns:
            (model): trained Logistic regression model
    """
    LogReg_model = LogisticRegression(max_iter=500, multi_class='ovr')
    LogReg_model.fit(train_X, train_Y)
    return LogReg_model

def train_SVM_model(train_X, train_Y, kernel):
    """
    Trains SVM model
    Args:
        train_X (list): list of tweets
        train_Y (list): list of classes
        kernel (str): the kernel to use
    Returns:
            (model): trained SVM model
    """
    if kernel == 'linear':
        SVM_model = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='linear'))
        SVM_model.fit(train_X, train_Y)
    else:
        SVM_model = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='sigmoid', C=10))
        SVM_model.fit(train_X, train_Y)
    return SVM_model

def train_NaiveBayes_model(train_X, train_Y):
    """
    Trains Naive bayes model
    Args:
        train_X (list): list of tweets
        train_Y (list): list of classes
    Returns:
            (model): trained Naive bayes model
    """
    NaiveBayes_model = MultinomialNB(fit_prior=False)
    NaiveBayes_model.fit(train_X, train_Y)
    return NaiveBayes_model

def train_FFNN_model(tweet_class_list, train_X_splitted, train_Y_splitted):
    """
    Trains FFNN model
    Args:
        tweet_class_list (list): list of tuples (pp tweet, class, original tweet)
        train_X_splitted (list): list of tweets
        train_Y_splitted (list): list of classes
    Returns:
            (model): trained FFNN model
            (dict): token vocabulary
            (device): the device the model ran on
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tweet_dict = make_dict([[token for token in tpl[0].split()] for tpl in tweet_class_list], False)

    input_dim = len(tweet_dict)
    hidden_dim = 500
    output_dim = 2
    num_epochs = 20

    ff_nn_bow_model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    ff_nn_bow_model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ff_nn_bow_model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for index, tweet in enumerate(train_X_splitted):
            optimizer.zero_grad()
            bow_vec = make_bow_vector(tweet_dict, tweet.split(), device)
            probs = ff_nn_bow_model(bow_vec)
            target = make_target(train_Y_splitted[index], device)
            loss = loss_function(probs, target)
            loss.backward()
            optimizer.step()

    return ff_nn_bow_model, tweet_dict, device


def predict_FFNN_model(ff_nn_bow_model,tweet_dict, test_X_splitted, test_Y_splitted, device):
    """
    Makes predictions with FFNN model:
    Args:
        ff_nn_bow_model (model): trained FFNN model
        tweet_dict (dict): token vocabulary
        test_X_splitted (list): tweets test set
        test_Y_splitted (list): classes test set
        device (device): the device the FFNN model ran on
    Return:
            (list): original test set predictions
            (list): the predictions made by the FFNN model
    """
    bow_ff_nn_predictions = []
    original_lables_ff_bow = []
    with torch.no_grad():
        for index, tweet in enumerate(test_X_splitted):
            bow_vec = make_bow_vector(tweet_dict, tweet.split(), device)
            probs = ff_nn_bow_model(bow_vec)
            bow_ff_nn_predictions.append(torch.argmax(probs, dim=1).cpu().numpy()[0])
            original_lables_ff_bow.append(make_target(test_Y_splitted[index], device).cpu().numpy()[0])
    return original_lables_ff_bow, bow_ff_nn_predictions



def separte_tweets(tweets_df):
    """
    Make a list of tuples contains pp tweet, class and original tweet
    Args:
        tweets_df (dataframe): dataframe after reading the tsv file
    Returns:
            (list): list of tuples
    """
    stop_words = set(stopwords.words('english'))
    tweet_class_list = []
    for index, row in tweets_df.iterrows():
        pp_tweet = preprocess(row['tweet_text'], stop_words)
        original_tweet = row['tweet_text']
        if row['user_handle'] == 'realDonaldTrump':
            device, tweet_date = row['device'], get_date(row['time_stamp'])
            change_device_date = get_date('2017-04-01')
            if device == 'android':
                if tweet_date < change_device_date:
                    tweet_class_list.append((pp_tweet, 0, original_tweet))
                else:
                    tweet_class_list.append((pp_tweet, 1, original_tweet))
            elif device == 'iphone':
                if tweet_date > change_device_date:
                    tweet_class_list.append((pp_tweet, 0, original_tweet))
                else:
                    tweet_class_list.append((pp_tweet, 1, original_tweet))
            else:
                tweet_class_list.append((pp_tweet, 1, original_tweet))
        elif row['user_handle'] == 'PressSec':
            tweet_class_list.append((pp_tweet, 1, original_tweet))
        elif row['user_handle'] == 'POTUS':
            time_tweeted = get_date(row['time_stamp'])
            trump_start, trump_end = [get_date(d) for d in ['2017-01-20','2021-01-20']]
            if trump_start <= time_tweeted <= trump_end:
                tweet_class_list.append((pp_tweet, 0, original_tweet))
            else:
                tweet_class_list.append((pp_tweet, 1, original_tweet))
    return tweet_class_list


def get_date(d):
    """
    Constructs a date object from date string
    Args:
        d (str): a date string
    Returns:
            (date): date object
    """
    if ':' in d:
        return datetime.strptime(d, '%Y-%m-%d %H:%M:%S').date()
    else:
        return datetime.strptime(d, '%Y-%m-%d').date()


def my_fit_transform(train_X_splitted, tokens):
    """
    Make 2D array of tweets and token features
    Args:
        train_X_splitted (list): a list of tweets
        tokens (set): vocabulary of tokens
    Returns:
            (np-array): 2D array
    """
    tweet_index = {tweet:index for index,tweet in enumerate(train_X_splitted)}
    token_index = {token:index for index,token in enumerate(tokens)}
    feature_arr = np.array([[0 for token in tokens] for tweet in train_X_splitted])
    for tweet in train_X_splitted:
        for token in tweet.split():
            feature_arr[tweet_index[tweet]][token_index[token]] += 1
    return feature_arr

def my_transform(test_X_splitted, tokens):
    """
    Make 2D array of tweets and token features
    Args:
        test_X_splitted (list): a list of tweets
        tokens (set): vocabulary of tokens
    Returns:
            (np-array): 2D array
    """
    tweet_index = {tweet: index for index, tweet in enumerate(test_X_splitted)}
    token_index = {token: index for index, token in enumerate(tokens)}
    feature_arr = np.array([[0 for token in tokens] for tweet in test_X_splitted])
    for tweet in test_X_splitted:
        for token in tweet.split():
            try:
                feature_arr[tweet_index[tweet]][token_index[token]] += 1
            except KeyError:
                continue
    return feature_arr



def get_length_feature(tweets):
    """
    Makes length feature vector from the tweets
    Args:
        tweets (list): list of tweets
    Returns:
            (np-array): 1D array of length feature
    """
    return np.array([[len(tweet.split())] for tweet in tweets])

def get_hashtag_feature(tweets):
    """
    Makes hashtag feature vector from the tweets
    Args:
        tweets (list): list of tweets
    Returns:
            (np-array): 1D array of hashtag feature
    """
    return np.array([[tweet.count('#')] for tweet in tweets])

def get_capitalletter_feature(tweets):
    """
    Makes capital letters feature vector from the tweets
    Args:
        tweets (list): list of tweets
    Returns:
            (np-array): 1D array of length feature
    """
    return np.array([[len([1 for token in tweet.split() if token[0].isupper()])] for tweet in tweets])

def form_features(twoD_array, original_tweets):
    """
    Makes 2D array from tweets with all features
    Args:
         twoD_array (np-array): 2D array of token features
         original_tweets (list): list of original tweets
    Returns:
            (np-array): 2D array of all features
    """
    features_arrays_train = [get_length_feature(original_tweets),
                             get_hashtag_feature(original_tweets),
                             get_capitalletter_feature(original_tweets)]
    for features in features_arrays_train:
        twoD_array = np.hstack((twoD_array, features))
    return twoD_array

def read_tsv(file_name, headers):
    """
    Reads tsv file
    Args:
        file_name (str): the name of the file
        headers (list): the headers of the table
    Returns:
            (dataframe): dataframe containg the tsv file
    """
    tsvfile = open(file_name,'r')
    tsvreader = csv.reader(tsvfile, delimiter = '\n')
    tweet_list = [line[0] for line in tsvreader]
    tweet_dict = {header:[] for header in headers}
    for tweet in tweet_list:
        splitted_tweet = tweet.split('\t')
        for part, header in zip(splitted_tweet,headers):
            tweet_dict[header].append(part)
    return pd.DataFrame(tweet_dict)


def preprocess(text, stop_words):
    """
    Preprocess input text
    Args:
        text (str): text to preprocess
        stop_words (list): list of stop-words
    Returns:
            (str): preprocessed text
    """
    lowered_text = text.lower()
    set_punctuations = {char for char in '''!()-[]{};:'"\,<>./?@#$%^&*_~'''}
    set_punctuations.add('\n')
    normalized_text = ''
    for index in range(len(lowered_text)):
        try:
            if lowered_text[index] not in set_punctuations:
                normalized_text += lowered_text[index]
            else:
                try:
                    if lowered_text[index+1] != ' ' and lowered_text[index-1] != ' ':
                        normalized_text += ' '
                except IndexError:
                    continue
        except UnicodeDecodeError:
            continue

    normalized_text = ' '.join([part for part in normalized_text.split() if part not in stop_words])
    return normalized_text

def split_train_test(XY_list):
    """
    Splits data to train and test sets
    Args:
        XY_list (list): list of tuples containing the data
    Returns:
            (tuple): four list of XY train and test
    """
    return train_test_split([(tpl[0], tpl[2]) for tpl in XY_list], [tpl[1] for tpl in XY_list], test_size=0.2 ,shuffle=True)

def use_cross_validation(tweet_class_list, model, folds,vectorizer):
    """
    Applys cross validation on train and test set with specific model
    Args:
         tweet_class_list (list): list of tuples
         model (model): the model to use cross validation on
         folds (int): how much folds
         vectorizer (vectorizer): vectorizer object for feature exraction
    Returns:
            (list): list of acccuracies
    """
    return cross_val_score(model, vectorizer.fit_transform([tpl[0] for tpl in tweet_class_list]).toarray(),
                    [str(value) for value in [tpl[1] for tpl in tweet_class_list]],
                    cv=folds)

def get_test_data(fn):
    """
    Reads the test data and applys preprocessing on it
    Args:
        fn (str): test file path
    Returns:
            (list): list of tuples of pp tweet and original tweet
    """
    stop_words = set(stopwords.words('english'))
    tweets_df_test = read_tsv(fn, ['user_handle','tweet_text','time_stamp'])
    tweet_list = []
    for index, row in tweets_df_test.iterrows():
        pp_tweet = preprocess(row['tweet_text'], stop_words)
        original_tweet = row['tweet_text']
        tweet_list.append((pp_tweet, original_tweet))
    return tweet_list

def write_test_predictions(predictions):
    """
    Writes test prediction to file
    Args:
         predictions (list): list of 0,1 predictions
    """
    result_string = ' '.join([str(pred) for pred in predictions])
    open('317649606.txt','w').write(result_string)


def make_dict(token_list, padding=True):
    """
    Makes dictionary from the dataset
    Args:
        token_list (list): list of tokens of the dataset
        padding (bool): whether to apply padding
    Returns:
            (dict): dictionary of the dataset
    """
    if padding:
        tweet_dict = corpora.Dictionary([['pad']])
        tweet_dict.add_documents(token_list)
    else:
        tweet_dict = corpora.Dictionary(token_list)
    return tweet_dict

def make_bow_vector(tweet_dict, sentence, device):
    """
    Makes bow vector from sentence (tweet)
    Args:
        tweet_dict (dict): dictionary containing tweets
        sentence (str): the tweet
        device (device): cpu/gpu
    Returns:
            (list): a bow vector
    """
    vec = torch.zeros(len(tweet_dict), dtype=torch.float64, device=device)
    for word in sentence:
        vec[tweet_dict.token2id[word]] += 1
    return vec.view(1, -1).float()

def make_target(label, device):
    """
    Makes a target vector
    Args:
        label (int): the class of the tweet
        device (device): cpu/gpu
    Returns:
            (list): a target vector
    """
    if label == 0:
        return torch.tensor([0], dtype=torch.long, device=device)
    elif label == 1:
        return torch.tensor([1], dtype=torch.long, device=device)


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return F.softmax(out, dim=1)



if __name__ == '__main__':
    main()