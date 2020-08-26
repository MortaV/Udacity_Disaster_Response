import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')


from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.externals import joblib

def load_data(database_filepath):
    """
    Loads the data from sqlite database and separates it to input and output variables and output names
    
    INPUT:
        database_filepath - path to the sqlite database
        
    OUTPUT:
        X - message column from the database
        Y - categories for the classification
        categories - names of the categories for the classification
    """
    # loading the data
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages', engine)
    
    # separating to input, output variables and output names
    X = df['message']
    Y = df.drop(['message', 'original', 'genre'], axis=1)
    categories = Y.columns

    return X, Y, categories

def tokenize(text):
    """
    Tokenizing the input text (together with lemmatizer, lowering the letters and stripping the text from unneeded spaces)
    
    INPUT:
        text - a string for tokenization
    
    OUTPUT:
        clean_tokens - a list of tokenized words
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens

def build_model():
    """
    Training the model (pipeline + parameters for GridSearchCV)
    
    OUTPUT:
        cv - output of GridSearchCV
    """
    # pipeline for the model
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # parameters for the Grid Search
    parameters = {
        'clf__estimator__n_estimators': [3, 5, 10, 15],
        'clf__estimator__max_depth': [None, 3, 5],
        'vect__stop_words': [None, stopwords.words('english')]
    }
    
    # Grid Search
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Printing the results of our machine learning algorithm. Each category from Y gets a separate line.
    
    INPUT:
        model - model object
        X_test - test input data
        Y_test - test output data
        category_names - names of the categories in Y_test
    
    OUTPUT:
        the output is only the printed results from classification_report
    """
    # converting Y predictions to the dataframe
    Y_pred = pd.DataFrame(model.predict(X_test))
    
    # print results for each category    
    i=0
    for column in category_names:
        print(column, '\n', classification_report(Y_test[column], Y_pred.iloc[:,i]))
        i+=1

def save_model(model, model_filepath):
    """
    Saving our model to a pickle format, in the provided path
    
    INPUT:
        model - model object that needs to be saved
        model_path - path where to save the pickle file
    
    OUTPUT:
        pickle file, saved in model_path
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()