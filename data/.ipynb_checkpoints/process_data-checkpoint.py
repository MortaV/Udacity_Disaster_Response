import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Combines messages and categories files to one
    
    INPUT:
        messages_filepath - path to the messages data file
        categories_filepath - path to the categories data file
        
    OUTPUT:
        df - dataframe with combined messages and categories files
    """
    messages = pd.read_csv(messages_filepath).set_index('id')
    categories = pd.read_csv(categories_filepath).set_index('id')
    df = pd.concat([messages, categories], axis=1)
    
    return df

def clean_data(df):
    """
    Cleans the output of load_data(): splits the categories and removes duplicated values
    
    INPUT:
        df - dataframe from the load_data() output
        
    OUTPUT:
        df - clean dataframe
    """
    # split categories into separate columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [x[-1] for x in categories[column]]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drops old categories column and adds cleaned ones
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    # remove duplicated values
    df = df[~df.duplicated()]
    
    return df

def save_data(df, database_filename):
    """
    Saving data to the sqlite database
    
    INPUT:
        df - dataframe from the clean_data() output
        database_filename - string, the name for the database
        
    OUTPUT:
        a file for sqlite created
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()