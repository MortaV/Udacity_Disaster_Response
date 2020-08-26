import json
import plotly
import pandas as pd
from plotly.graph_objects import Bar, layout, Layout
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # data for the distribution graph
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # data for the subplots graph
    genre_unique = df['genre'].unique()
    plotting_df = pd.melt(df, id_vars=['genre'], value_vars=df.columns[3:])
    plotting_df = plotting_df.groupby(['genre', 'variable']).sum().reset_index()
    
    # graph number 2
    fig1 = make_subplots(rows=genre_unique.shape[0], cols=1, print_grid=False, subplot_titles=genre_unique)
    
    i=1
    for genre in genre_unique:
        data=plotting_df[plotting_df['genre']==genre]
        fig1.add_trace(Bar(x=data['variable'], y=data['value'], opacity=0.5, marker=dict(color='#F1C40F')), row=i, col=1)
        i+=1
    
    # cleaning the layout of the graphs
    layout_custom = layout.Template(
        layout=Layout(titlefont=dict(size=24, color='#34495E'))
    )
    
    fig1['layout'].update(title='Messages by genre and category', 
                      showlegend=False,
                      template=layout_custom)
    
    fig1['layout']['yaxis1'].update(hoverformat=',d', tickformat=',d')
    fig1['layout']['yaxis2'].update(hoverformat=',d', tickformat=',d')
    fig1['layout']['yaxis3'].update(hoverformat=',d', tickformat=',d')
    fig1['layout']['xaxis1'].update(visible=False)
    fig1['layout']['xaxis2'].update(visible=False)
    
    # graph number 1
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    opacity=0.5, 
                    marker=dict(color='#F1C40F')
                )
            ],

            'layout': {
                'template': layout_custom,
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }]
    
    graphs.append(fig1)
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()