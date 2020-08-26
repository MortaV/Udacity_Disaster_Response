# Udacity Disaster Response

1. [Summary](#summary)
2. [How to use it](#how-to-use-it)
2. [Main contacts](#main-contacts)
3. [Where is everything stored](#where-is-everything-stored)
4. [Requirements](#requirements)
5. [Main files](#main-files)
6. [Results](#results)

## Summary

The goal of this project is to build an app that can be used with disaster response messages. This is a project from Udacity course for Data Science.

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

![ScreenShot](/screenshots/data analysis.png)

![ScreenShot](/screenshots/messages.png)

## Main contacts

Client side:

| Client name | Email  | Role in the project |
| :--- | :---: | :---: |
| Name Surname | email@email.com | Ad manager |

Internal:

| Name | Email  | Role in the project |
| :--- | :---: | :---: |
| Name Surname | email@email.com | Project manager |

## Where is everything stored

Graphs: Link to Teams or Google Drive repository <br>
Data: Kindred database (connector is in the code, credentials are needed)

## Requirements

Python packages:
- chart_studio==1.0.0
- pandas==1.0.1
- plotly==4.5.2
- numpy==1.18.1
- scipy==1.3.2
- statsmodels==0.11.1

## Main files

- **.gitignore** - file with the list of directories/files that should be not pushed to git with git push.
- **README.md** - main information about the projects
- **requirements.txt** - file with required Python projects

## Results

Add a short summary of the results of your investigation or project.


