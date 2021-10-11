#Pre requisite : !conda install -c plotly plotly-orca
#Pre requisite : !pip install torch
#Pre requisite : !pip install statistics
import re
import os
import torch
import statistics
import tweepy
import plotly.express as px
import numpy as np
import plotly.express as px
import pandas as pd
import torch.nn.functional as F
from model import TransformerWithClfHeadAndAdapters
from pytorch_transformers import BertTokenizer, cached_path

def loadedmodel():
    model_path = "models/transformer"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = torch.load(cached_path(os.path.join(model_path, "model_training_args.bin")))
    model = TransformerWithClfHeadAndAdapters(config["config"],
                                              config["config_ft"]).to(device)
    state_dict = torch.load(cached_path(os.path.join(model_path, "model_weights.pth")),
                            map_location=device)

    model.load_state_dict(state_dict)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    clf = tokenizer.vocab['[CLS]']  # classifier token
    pad = tokenizer.vocab['[PAD]']  # pad token
    max_len = config['config'].num_max_positions
    return model,tokenizer,clf,pad,max_len,device

def clean_tweet(text):
    text = re.sub(r'@\S+',r'', text)
    # text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    clean_tweet = ""
    for word in text:
        clean_tweet = clean_tweet + word
    return clean_tweet

def twitter_handle(twitter_link):
    # Fetch the handle/ screen name from the twitter link
    start = re.sub('https://www.twitter.com/','',twitter_link)

    handle = re.sub('\?.*','',start)
    return handle

def twitter_authentication():
    consumer_key = "key"
    consumer_secret = "secret"
    access_token = "token"
    access_token_secret = "access_token_secret"
    # Creating the authentication object
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    # Setting your access token and secret
    auth.set_access_token(access_token, access_token_secret)
    # Creating the API object while passing in auth information
    api = tweepy.API(auth)
    #Return the authentication reference point
    return api

def gettweets(api,name):
    api = api
    tweetCount = 200
    id = name
    # Calling the user_timeline function with our parameters
    results = api.user_timeline(id=id, count=tweetCount, tweet_mode='extended')
    allTweets = []
    for tweet in results:
        allTweets.append({'tweet': tweet.full_text})
    pd.set_option("display.max_colwidth", None)
    new = pd.DataFrame.from_dict(allTweets)
    new.to_csv('tweets.csv')
    new['tweets'] = new['tweet'].apply(clean_tweet)
    wordss = new['tweets']
    return wordss, new
def calculate(cleaned_tweet_list):
    def bertencoder(inputs):
        return list(tokenizer.convert_tokens_to_ids(o) for o in inputs)
    model,tokenizer,clf,pad,max_len,device = loadedmodel()
    round_off_list=[]
    encoded_texts=[]
    for i in range(len(cleaned_tweet_list)):
        text = cleaned_tweet_list[i]
        inputs = tokenizer.tokenize(text)
        if len(inputs) >= max_len:
            inputs = inputs[:max_len - 1]
        ids = bertencoder(inputs) + [clf]
        encoded_texts.append(ids)
    logits_list=[]
    for i in range(len(encoded_texts)):
        encoded_text = encoded_texts[i]
        with torch.no_grad():
            tensor = torch.tensor(encoded_text, dtype=torch.long).to(device)
            tensor_reshaped = tensor.reshape(1, -1)
            tensor_in = tensor_reshaped.transpose(0, 1).contiguous()
            logits = model(tensor_in,
                           clf_tokens_mask=(tensor_in == clf),
                           padding_mask=(tensor_reshaped == pad))
        logits_list.append(logits)

    soft_max_list=[]
    for i in range(len(logits_list)):
        logits_tweet=logits_list[i]
        val, _ = torch.max(logits_tweet, 0)
        val = F.softmax(val, dim=0).detach().cpu().numpy()
        soft_max_list.append(val)
    for i in range(len(soft_max_list)):
        round_off_tweet=[]
        soft_max_tweet = soft_max_list[i]
        for x in range(len(soft_max_tweet)):
            round_off_tweet.append(f'{soft_max_tweet[x]:.5f}')
        round_off_list.append(round_off_tweet)
    return round_off_list
def main(name):
    api = twitter_authentication()
    tweet_main, tweet_dataframe = gettweets(api,name)
    tweet_sentiment_values = calculate(tweet_main)
    tweet_dataframe['Sentiment_Values'] = tweet_sentiment_values
    tweet_dataframe['tweets'].replace('', np.nan, inplace=True)
    tweet_dataframe['tweets'].replace(' ', np.nan, inplace=True)
    tweet_dataframe.dropna(subset=['tweets'], inplace=True)
    tweet_dataframe = tweet_dataframe.reset_index(drop=True)

    sentiment_values_dataframe = tweet_dataframe['Sentiment_Values']
    strongly_negative_values = []
    weakly_negative_values = []
    neutral_values = []
    weakly_positive_values = []
    strongly_positive_values = []
    mean_polarity = []
    for x in range(len(sentiment_values_dataframe)):
        strongly_negative_values.append(float(sentiment_values_dataframe[x][0]))

    for x in range(len(sentiment_values_dataframe)):
        weakly_negative_values.append(float(sentiment_values_dataframe[x][1]))

    for x in range(len(sentiment_values_dataframe)):
        neutral_values.append(float(sentiment_values_dataframe[x][2]))

    for x in range(len(sentiment_values_dataframe)):
        weakly_positive_values.append(float(sentiment_values_dataframe[x][3]))

    for x in range(len(sentiment_values_dataframe)):
        strongly_positive_values.append(float(sentiment_values_dataframe[x][4]))
    tweet_dataframe['strongly negative'] = strongly_negative_values
    tweet_dataframe['weakly negative'] = weakly_negative_values
    tweet_dataframe['neutral'] = neutral_values
    tweet_dataframe['weakly positive'] = weakly_positive_values
    tweet_dataframe['strongly positive'] = strongly_positive_values

    mean_polarity.append(statistics.mean(strongly_negative_values))
    mean_polarity.append(statistics.mean(weakly_negative_values))
    mean_polarity.append(statistics.mean(neutral_values))
    mean_polarity.append(statistics.mean(weakly_positive_values))
    mean_polarity.append(statistics.mean(strongly_positive_values))
    sentiment_values_dataframe = pd.DataFrame(dict(r=mean_polarity,theta=['Strongly Negative', 'Weakly Negative', 'Weutral', 'Weekly Positive', 'Strongly Positive']))
    fig = px.line_polar(sentiment_values_dataframe, r='r', theta='theta', line_close=True)
    fig.update_traces(fill='toself')
    fig.show()
    fig.write_image("sentiment_radar_chart.png")
    return mean_polarity
    
def sentence_sentiment(sentence):
    sentences = []
    sentences.append({'initial_sentence': sentence})
    pd.set_option("display.max_colwidth", None)
    sentence_dataframe = pd.DataFrame.from_dict(sentences)
    sentence_dataframe['cleaned_sentence'] = sentence_dataframe['initial_sentence'].apply(clean_tweet)
    sentence_for_sentiment = sentence_dataframe['cleaned_sentence']
    sentence_sentiment_values = calculate(sentence_for_sentiment)
    sentence_dataframe['Sentiment_Values'] = sentence_sentiment_values
    sentence_dataframe['cleaned_sentence'].replace('', np.nan, inplace=True)
    sentence_dataframe['cleaned_sentence'].replace(' ', np.nan, inplace=True)
    sentence_dataframe.dropna(subset=['cleaned_sentence'], inplace=True)
    sentence_dataframe = sentence_dataframe.reset_index(drop=True)

    sentiment_values_dataframe_sentence = sentence_dataframe['Sentiment_Values']
    strongly_negative_values = []
    weakly_negative_values = []
    neutral_values = []
    weakly_positive_values = []
    strongly_positive_values = []
    mean_polarity_for_sentence = []
    for x in range(len(sentiment_values_dataframe_sentence)):
        strongly_negative_values.append(float(sentiment_values_dataframe_sentence[x][0]))

    for x in range(len(sentiment_values_dataframe_sentence)):
        weakly_negative_values.append(float(sentiment_values_dataframe_sentence[x][1]))

    for x in range(len(sentiment_values_dataframe_sentence)):
        neutral_values.append(float(sentiment_values_dataframe_sentence[x][2]))

    for x in range(len(sentiment_values_dataframe_sentence)):
        weakly_positive_values.append(float(sentiment_values_dataframe_sentence[x][3]))

    for x in range(len(sentiment_values_dataframe_sentence)):
        strongly_positive_values.append(float(sentiment_values_dataframe_sentence[x][4]))
    sentence_dataframe['strongly negative'] = strongly_negative_values
    sentence_dataframe['weakly negative'] = weakly_negative_values
    sentence_dataframe['neutral'] = neutral_values
    sentence_dataframe['weakly positive'] = weakly_positive_values
    sentence_dataframe['strongly positive'] = strongly_positive_values

    mean_polarity_for_sentence.append(statistics.mean(strongly_negative_values))
    mean_polarity_for_sentence.append(statistics.mean(weakly_negative_values))
    mean_polarity_for_sentence.append(statistics.mean(neutral_values))
    mean_polarity_for_sentence.append(statistics.mean(weakly_positive_values))
    mean_polarity_for_sentence.append(statistics.mean(strongly_positive_values))
    sentiment_values_dataframe_sentence = pd.DataFrame(dict(r=mean_polarity_for_sentence,theta=['Strongly Negative', 'Weakly Negative', 'Weutral', 'Weekly Positive', 'Strongly Positive']))
    fig = px.line_polar(sentiment_values_dataframe_sentence, r='r', theta='theta', line_close=True)
    fig.update_traces(fill='toself')
    fig.show()
    fig.write_image("sentiment_radar_chart.png")
    return mean_polarity_for_sentence