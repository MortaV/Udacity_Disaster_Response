# Udacity Disaster Response

1. [Summary](#summary)
2. [How to use it](#how-to-use-it)
3. [Main files](#main-files)
6. [Acknowledgements](#acknowledgements)

## Summary

The goal of this project is to build an app that can be used with disaster response messages. This is a project from Udacity course for Data Science.

All codes are stored on https://github.com/MortaV/Udacity_Disaster_Response.

## How to use it

### Instructions for setting up the app:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Instructions for using the app:

The app has 2 graphs on the first page (please see the screenshot below).

![ScreenShot](/screenshots/data_analysis.png)

You need to put a message you want to classify to the field (1) and press "Classify Message" (3). You will see the results instead of the graphs (3). If the genre is filled with green, it means the message is classified like that.

![ScreenShot](/screenshots/messages.png)

## Main files

*app* folder:

- **run.py** - main script for running the app. You can find codes for the graphs in there.
- **templates** - folder with the files for the UI of the app.

*data* folder:

- **disaster_categories.csv** - data for the categories.
- **disaster_messages.csv** - data with messages.
- **DisasterResponse.db** - database created with process_data.py.
- **process_data.py** - python scripts for cleaning, joining and pushing the data to the database.

*models* folder:

- **train_classifier.py** - python scripts for training the model on the data from the DisasterResponse.db. 
- **classifier.pkl** - NOT INCLUDED. The file is too big to store it on GitHub, but this is hgow the file created by train_classifier.py will be named.

## Acknowledgements

This is a project from Udacity Data Science nanodegree.