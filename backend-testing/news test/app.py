# simple of using flask as an backend for html to see the result you should 
# run this code and open the corresponded html is the same folder
# the result of the predection will print in the html screen

from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from textblob import TextBlob
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def scrape_news(company_name):
    url = f"https://news.google.com/search?q={company_name.replace(' ', '%20')}%20Saudi%20Arabia"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    headlines = soup.find_all('a', {'class': 'DY5T1d'})
    articles = [headline.get_text() for headline in headlines]
    return articles[:5]

def analyze_sentiment(headlines):
    sentiments = []
    for headline in headlines:
        analysis = TextBlob(headline)
        polarity = analysis.sentiment.polarity
        if polarity > 0:
            sentiments.append("Positive")
        elif polarity < 0:
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")
    return sentiments

def make_recommendation(sentiments):
    positive = sentiments.count("Positive")
    negative = sentiments.count("Negative")
    neutral = sentiments.count("Neutral")
    if positive > negative:
        return "Buy"
    elif negative > positive:
        return "Sell"
    else:
        return "Hold"

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    company_name = data.get("company_name")
    articles = scrape_news(company_name)
    sentiments = analyze_sentiment(articles)
    recommendation = make_recommendation(sentiments)
    return jsonify({"articles": articles, "sentiments": sentiments, "recommendation": recommendation})

if __name__ == "__main__":
    app.run(debug=True)
