import requests

from transformers import pipeline

API_KEY = open('API_KEY').read()

print(API_KEY[:5])

keyword = 'gold'
date = '2025-02-17'

pipe = pipeline("text-classification", model="ProsusAI/finbert")

url = (
    'https://newsapi.org/v2/everything?'
    f'q={keyword}&'
    f'from={date}&'
    'sortBy=popularity&'
    f'apiKey={API_KEY}'
)

response = requests.get(url)

articles = response.json()['articles']
articles = [article for article in articles if keyword.lower() in article['title'].lower() or keyword.lower() in article['description'].lower()]

total_score = 0
num_articles = 0

for i, article in enumerate(articles):

    print(f'Title: {article["title"]}')
    print(f'Link: {article["url"]}')
    print(f'Description: {article["description"]}')

    sentiment = pipe(article['content'])[0]

    print(f'Sentiment {sentiment["label"]}, Score: {sentiment["score"]}')
    print('-' * 40)

    if sentiment['label'] == 'positive':
        total_score += sentiment['score']
        num_articles += 1
    elif sentiment['label'] == 'negative':
        total_score -= sentiment['score']
        num_articles += 1

final_score = total_score / num_articles
print(f'Overall Sentiment: {"Positive" if final_score >= 0.15 else "Negative" if final_score <= -0.15 else "Neutral"} {final_score}')