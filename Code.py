# Registration Number: 1908054

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.stem import WordNetLemmatizer

NO_CONTENT_TITLE = False # set to 'True' for entire title samples
IDF = True # set to 'True' for using idf with tf weight
LEMMA = True # True for using Lemmatizer
LOWERCASE = True # True for using lowercase
PATH = "econbiz.csv" # data set path
NGRAM = True # if this is set to True, n-gram range below need to be define (>1)
n_range = 2

classifier = LogisticRegression()  # ML classifier as Logistic Regression (LR)
binary_relevance = OneVsRestClassifier(classifier, n_jobs=4)  # binary relevance with LR

# define vecterizer according to the setting above
if not NGRAM:
    vectorizer = TfidfVectorizer(max_features=25000, lowercase=LOWERCASE, use_idf=IDF)  # unigram
else:
    vectorizer = TfidfVectorizer(lowercase=LOWERCASE, use_idf=IDF, ngram_range=(1, n_range))  # n-gram

# use Pipeline module to join preprocessing output with classifier
clf = Pipeline([("vectorizer", vectorizer), ("classifier", binary_relevance)])

# define lemmtizer callable object
wordnet_lemmatizer = WordNetLemmatizer()


# input function used to read the data samples based on the setting
def input(path, NO_CONTENT_TITLE=False):
    dataframe = pd.read_csv(path) # read CSV data
    if not NO_CONTENT_TITLE:
        dataframe = dataframe[dataframe["fold"].isin(range(0, 10))] # 10-fold splitting
    return dataframe


def process_label(labelset, data):
    label_set = labelset["labels"].values
    labels = data["labels"].values  # store value of label column from read data
    labels = [[lb for lb in lb_str.split()] for lb_str in labels]  # store list of label in to labels variable
    multilabel_binarizer = MultiLabelBinarizer()  # define callable MultiLabelBinarizer object
    multilabel_binarizer.fit(labels)  # use MultiLabelBinarizer object to fit on the label data
    label_set = [[lb for lb in lb_str.split()] for lb_str in label_set]
    return multilabel_binarizer.transform(label_set)


def split_train_test(data, fold_i):
    df = data

    test_df = df[df["fold"] == fold_i] # store test sample by fold indicator (index)
    X_test = test_df["title"].values # store test sample by fold indicator (text)
    if LEMMA:
        c = 0
        for i in X_test:
            split_word = i.split() # break text in to word
            temp_str = ""
            for j in split_word:
                temp_str += wordnet_lemmatizer.lemmatize(j) # apply lemmatizer to text
                temp_str += " "
            X_test[c] = temp_str # replace text with pre-processed text
            # print(i)
            # print(X_test[c])
            c += 1
    y_test = process_label(test_df, data) # encoder test label vaule with process_label function

    train_df = df[df["fold"] != fold_i] # store train sample by fold indicator (index)
    X_train = train_df["title"].values # store train sample by fold indicator (text)
    if LEMMA:
        c = 0
        for i in X_train:
            split_word = i.split()  # break text in to word
            temp_str = ""
            for j in split_word:
                temp_str += wordnet_lemmatizer.lemmatize(j)  # apply lemmatizer to text
                temp_str += " "
            X_train[c] = temp_str  # replace text with pre-processed text
            # print(i)
            # print(X_train[c])
            c += 1
    y_train = process_label(train_df, data) # encoder train label vaule with process_label function
    return X_train, y_train, X_test, y_test


def evaluation(data):
    temp = 0.0 # initialize the  f-1 score
    for fold in range(0, 10): # loop ten time to  train and calculate f-1 score with different train/test based on
        # fold member
        train_df, y_train, test_df, y_test = split_train_test(data, fold) # call to get train and test samples of the
        # current fold
        clf.fit(train_df, y_train) # train classifier
        y_pred = clf.predict(test_df) # predict test labels
        temp += f1_score(y_test, y_pred, average="samples") # calculate f-1 score
        if not NO_CONTENT_TITLE:
            break
    if NO_CONTENT_TITLE:
        temp /= 10 # average f-1 score fold stamp
    return temp


data = input(PATH) # read input data
print("sample-average f-1 score:", evaluation(data)) # display the evaluation result
