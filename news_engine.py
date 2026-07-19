import requests
import feedparser
import urllib.parse
from config import NEWS_API_KEY

def get_news(stock):
    headlines = []

    # NewsAPI
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": stock,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY
    }

    try:
        r = requests.get(url, params=params, timeout=10).json()
        if "articles" in r:
            for article in r["articles"][:5]:
                headlines.append(article["title"])
    except Exception as exc:
        print(f"NewsAPI fetch failed for {stock}: {exc}")

    # Google RSS
    try:
        rss_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(stock)}"
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:5]:
            headlines.append(entry.title)
    except Exception as exc:
        print(f"Google RSS fetch failed for {stock}: {exc}")

    return headlines