# sentiment_analysis.py

from textblob import TextBlob

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # Returns a value between -1 and 1
    
    if sentiment > 0:
        sentiment_result = "Positive"
    elif sentiment < 0:
        sentiment_result = "Negative"
    else:
        sentiment_result = "Neutral"
    
    return sentiment_result
