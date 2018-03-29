'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn import neighbors 
from sklearn.cross_validation import cross_val_score 
from sklearn.ensemble import RandomForestClassifier  
from sklearn.grid_search import GridSearchCV 
from sklearn.naive_bayes import GaussianNB 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    loss_train = mean_squared_error(train_pred, train_labels)
    print('BernoulliNB baseline train loss =',loss_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    loss_test = mean_squared_error(test_pred, test_labels)
    print('BernoulliNB baseline test loss =',loss_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def rf(bow_train, train_labels, bow_test, test_labels):
    rf = RandomForestClassifier(n_estimators=500, max_depth=500, min_samples_leaf = 2)
    rf.fit(bow_train, train_labels) 
    train_pred = rf.predict(bow_train)
    loss_train = mean_squared_error(train_pred, train_labels)
    print('rf regression train loss =',loss_train)
    print('rf train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = rf.predict(bow_test)
    loss_test = mean_squared_error(test_pred, test_labels)
    print('rf test loss =',loss_test)
    print('rf test accuracy = {}'.format((test_pred == test_labels).mean()))
    
    return rf

def rf_parameter(bow_train, train_labels):
# Set the parameters by cross-validation
    parameter_space = {
        "n_estimators": [50,100,500,1000],
        "max_depth": [50, 100, 500, 1000],
        "min_samples_leaf": [1, 2, 3, 4],
    }
    
    
    print("# Tuning hyper-parameters for accuracy")
    print()
    
    clf = RandomForestClassifier(random_state=14)
    grid = GridSearchCV(clf, parameter_space, cv=5, scoring='accuracy')
    grid.fit(bow_train, train_labels)
    print("Best parameters set found on development set:")
    print(grid.best_params_)
    
    
def gnb(bow_train, train_labels, bow_test, test_labels):
    nb = GaussianNB()  
    bow_train = bow_train.todense()
    bow_test = bow_test.todense()
    nb.fit(bow_train, train_labels)
    train_pred = nb.predict(bow_train)
    loss_train = mean_squared_error(train_pred, train_labels)
    print('gaussian naive bayes train loss =',loss_train)
    print('gaussian naive bayes train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = nb.predict(bow_test)
    loss_test = mean_squared_error(test_pred, test_labels)
    print('gaussian naive bayes test loss =',loss_test)
    print('gaussian naive bayes test accuracy = {}'.format((test_pred == test_labels).mean()))
    
    return nb


def logreg(bow_train, train_labels, bow_test, test_labels):
    logreg = linear_model.LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial')  
    logreg.fit(bow_train, train_labels)  
    train_pred = logreg.predict(bow_train)
    loss_train = mean_squared_error(train_pred, train_labels)
    print('logistic regression train loss =',loss_train)
    print('logistic regression train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = logreg.predict(bow_test)
    loss_test = mean_squared_error(test_pred, test_labels)
    print('logistic regression test loss =',loss_test)
    print('logistic regression test accuracy = {}'.format((test_pred == test_labels).mean()))
    
    return logreg
    
def log_parameter(bow_train, train_labels):
# Set the parameters by cross-validation
    parameter_space = {
        "C": [1,10,100,1000],
        "solver": ['lbfgs'], ##'newton-cg':slow    'sag':not converge
    }
    
    
    print("# Tuning hyper-parameters for accuracy")
    print()
    
    clf = linear_model.LogisticRegression()
    grid = GridSearchCV(clf, parameter_space, cv=5, scoring='accuracy')
    grid.fit(bow_train, train_labels)
    print("Best parameters set found on development set:")  
    print(grid.best_params_)
    
def confuse_matrix(model,bow_test, test_labels):
    test_pred = model.predict(bow_test)
    matrix = np.zeros((20,20))
    for i in range(len(test_pred)):
        matrix[test_pred[i],test_labels[i]] += 1 
    find_class = matrix - 10000000*np.eye(20,20)    ##making the diagnoal elements small for not being picked up
    itemindex = np.argwhere(find_class == np.max(find_class))
    print("most confused classes are",itemindex)
    return matrix
    
    
    
        
train_data, test_data = load_data()
train_bow, test_bow, feature_names = bow_features(train_data, test_data)

bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)
##rf_parameter(train_bow, train_data.target)
rf_model = rf(train_bow, train_data.target, test_bow, test_data.target)
nb_model = gnb(train_bow, train_data.target, test_bow, test_data.target)
##log_parameter(train_bow, train_data.target)
log_model = logreg(train_bow, train_data.target, test_bow, test_data.target)
confuse_matrix = confuse_matrix(rf_model,test_bow,test_data.target)