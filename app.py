import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt



st.title("Sentiment Analysis of News Headlines about Company")
st.sidebar.title("Sentiment Analysis of News-Headline")
st.markdown("This dashboard is used to analyze sentiments of News Headlines about a stock company")
# st.sidebar.markdown("This application is a Streamlit dashboard used "
#             "to analyze sentiments of tweets 🐦")


@st.cache(persist=True)
def load_data():
    data = pd.read_csv('finviz_tickers.csv')
    return data

data = load_data()

import spacy
nlp = spacy.load('en_core_web_sm')
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from urllib.request import urlopen, Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
pd.set_option('max_rows',999)
# Parameters 
n = 3 #the # of article headlines displayed per ticker
# tickers = ['TSLA']

st.sidebar.markdown("### Stock Ticker Selection")
# select = st.sidebar.selectbox('Select ticker', data['Ticker'].unique(), key='1')
ticker_value = st.sidebar.text_input('Select ticker','TSLA')
ticker_value  =[ticker_value]
print(ticker_value)

if ticker_value not in load_data()['Ticker'].unique():
	raise Exception('Stock Ticker Not found.Please put some valid stock ticker name')

st.sidebar.markdown("### Company Selection")
company_value = st.sidebar.text_input('Select ticker','Tesla')
company_value  =[company_value]
print(company_value)


def get_news_data():
    finwiz_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}

    for ticker in ticker_value:
        url = finwiz_url + ticker
        req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
        resp = urlopen(req)    
        html = BeautifulSoup(resp, features="lxml")
        news_table = html.find(id='news-table')
        news_tables[ticker] = news_table

    try:
        for ticker in ticker_value:
            df = news_tables[ticker]
            df_tr = df.findAll('tr')
        
            # print ('\n')
            # print ('Recent News Headlines for {}: '.format(ticker))
            
            for i, table_row in enumerate(df_tr):
                a_text = table_row.a.text
                td_text = table_row.td.text
                td_text = td_text.strip()
                # print(a_text,'(',td_text,')')
                if i == n-1:
                    break
    except KeyError:
        print('DO not')

    # Iterate through the news
    parsed_news = []
    for file_name, news_table in news_tables.items():
        for x in news_table.findAll('tr'):
            text = x.a.get_text() 
            date_scrape = x.td.text.split()

            if len(date_scrape) == 1:
                time = date_scrape[0]
                
            else:
                date = date_scrape[0]
                time = date_scrape[1]

            ticker = file_name.split('_')[0]
            
            parsed_news.append([ticker, date, time, text])
            
    # Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()

    columns = ['Ticker', 'Date', 'Time', 'Headline']
    news = pd.DataFrame(parsed_news, columns=columns)
    scores = news['Headline'].apply(analyzer.polarity_scores).tolist()

    df_scores = pd.DataFrame(scores)
    news = news.join(df_scores, rsuffix='_right')

    # print('asd',news)
    # View Data 
    news['Date'] = pd.to_datetime(news.Date).dt.date

    unique_ticker = news['Ticker'].unique().tolist()
    news_dict = {name: news.loc[news['Ticker'] == name] for name in unique_ticker}

    values = []
    for ticker in ticker_value: 
        dataframe = news_dict[ticker]
        dataframe = dataframe.set_index('Ticker')
        dataframe = dataframe.drop(columns = ['Headline'])
        # print ('\n')
        # print (dataframe.head())
        
        mean = round(dataframe['compound'].mean(), 2)
        values.append(mean)
        
    df = pd.DataFrame(list(zip(ticker_value[0], values)), columns =['Ticker', 'Mean Sentiment']) 
    df = df.set_index('Ticker')
    df = df.sort_values('Mean Sentiment', ascending=False)
    # print ('\n')
    # print (df)
    # del news['neu']
    news['timestamp']=news['Date'].astype('str')+' '+news['Time']
    del news['Date']
    del news['Time']

    ##Remove Unneecasry news headline
    headline=[]
    text=[]
    org=[]
    for i in range(100):
        piano_class_doc = nlp(news['Headline'][i])
        for ent in piano_class_doc.ents:
            headline.append(news['Headline'][i])
            text.append(ent.text)
            org.append(ent.label_)
    out=pd.DataFrame(headline)
    out['text']=text
    out['label']=org

    filtered_stock_news=out[out['label']=='ORG'].reset_index(drop=True).drop_duplicates(0)
    filtered_stock_news=filtered_stock_news.reset_index(drop=True)
    filtered_stock_news.columns=['Headline','text','label']

    ##Filter based on company name
    filtered_stock_news=filtered_stock_news[filtered_stock_news['Headline'].str.contains(company_value[0])].reset_index(drop=True)
    return filtered_stock_news,news


# Sentiment Analysis
def get_sentiment_analysis():
    analyzer = SentimentIntensityAnalyzer()
    filtered_stock_news=get_news_data()[0]
    # print('filtered_stock_news',filtered_stock_news.head())
    news=get_news_data()[1]
    print('news',news.shape)
    # print('filtered_stock_news',filtered_stock_news.shape)
    
    scores = filtered_stock_news['Headline'].apply(analyzer.polarity_scores).tolist()
    df_scores = pd.DataFrame(scores)
    filtered_stock_news = filtered_stock_news.join(df_scores, rsuffix='_right')
    del filtered_stock_news['text']
    del filtered_stock_news['label']
    # print('outnews',filtered_stock_news.head())
    try:
    	filtered_stock_news=filtered_stock_news.merge(news,on=['Headline','neg','pos','neu','compound'])
    except:
    	raise Exception('Stock Ticker not found.Please put some valid Stock Ticker')
    # print('before remove 0',get_news_data()[1]['compound'].describe())
    print('filtered_stock_news shape',filtered_stock_news.shape)
    # print(out_news[['Headline','compound']])
    filtered_stock_news_good_news=filtered_stock_news[filtered_stock_news['compound']!=0.0000]
    filtered_stock_news_news=filtered_stock_news_good_news[filtered_stock_news_good_news['timestamp']>'2020-05-01']

    print('filtered_stock_news_good_news after removing 0 and >date',filtered_stock_news_good_news.shape)
    filtered_stock_news_good_news=filtered_stock_news_good_news.sort_values('compound',ascending=True).reset_index(drop=True)
    return filtered_stock_news_good_news,filtered_stock_news


def get_headlines_with_sentiment():
    data=get_sentiment_analysis()[0]
    out_news_good_news=data.rename(columns={'compound':'Sentiment'})
    return out_news_good_news[['timestamp','Ticker','Headline','Sentiment']]

def get_headlines_pie():
    data=get_sentiment_analysis()[1]
    data['Sentiment']=data['compound'].map(lambda x: 'Negative' if x<0 else 'Neutral' if x==0 else 'Positive')
    return data

st.sidebar.markdown("### Number of News by Sentiment")
select = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='1')
pie_data=get_headlines_pie()
sentiment_count = pie_data['Sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})
if not st.sidebar.checkbox("Hide Graph.", False):
    st.markdown("### Number of News by Sentiment for "+ company_value[0])
    if select == 'Bar plot':
        fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
        st.plotly_chart(fig)




st.sidebar.markdown("### Sentiment analysis of News Headlines")
sentiment_df = get_headlines_with_sentiment()
sentiment_df=sentiment_df.sort_values('timestamp')

if not st.sidebar.checkbox("Hide Graph", False):
    st.markdown("### Sentiment analysis of News Headlines for " + company_value[0])
    fig = px.line(sentiment_df, x='timestamp', y='Sentiment',height=500)
    fig.update_xaxes(tickangle=90, tickfont=dict(family='Rockwell', color='crimson', size=14))

    st.plotly_chart(fig)

st.sidebar.markdown("### Stock Price Graph")
if not st.sidebar.checkbox("Hide Stock Graph", False):
    st.markdown("### Stock Price Graph for " +company_value[0])
import plotly.graph_objects as go
import yfinance as yf
import pandas_datareader as pdr
import pandas as pd

msft = yf.Ticker(ticker_value[0])
df_ori = msft.history('8d',interval='1d')
df_ori=df_ori.reset_index()
import plotly.graph_objects as go
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_ori['Date'], y=df_ori['Close'],line=dict(color='red', width=4,dash='dot'),
                    mode='lines+markers',
                    name='Stock Price'))
st.plotly_chart(fig2)


st.sidebar.markdown("### News Headlines with Sentiment")
sentiment_df = get_headlines_with_sentiment()
sentiment_df=sentiment_df.sort_values('timestamp').reset_index(drop=True)

# if not st.sidebar.checkbox("Hide News", False):
# 	st.map(modified_data)
#     if st.sidebar.checkbox("Show raw data", False):
#         st.write(modified_data)

st.markdown("### News Headlines for " + company_value[0])
# st.map(sentiment_df)
if st.sidebar.checkbox("Show news data", True):
    st.write(sentiment_df)


import streamlit as st
import base64
from io import BytesIO

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1',index = False)
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="news-headline.xlsx">Download News-Headline file</a>' # decode b'abc' => abc

st.markdown(get_table_download_link(sentiment_df), unsafe_allow_html=True)
