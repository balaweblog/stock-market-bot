import requests
import feedparser
import urllib.parse
from config import NEWS_API_KEY


def get_news(stock):
    """
    Pulls recent headlines for `stock` from NewsAPI and Google News RSS.

    Returns a dict:
      {
        "headlines": [...],       # combined headlines, possibly empty
        "available": bool,        # False only if EVERY source raised an
                                   # exception/error -- i.e. we have no
                                   # signal at all, as opposed to genuinely
                                   # zero headlines from working sources.
        "sources_failed": [...],  # which named sources errored, for logging
      }

    Previously this returned a bare list, so a total fetch failure (both
    NewsAPI and RSS erroring out) looked identical to "no news today" --
    both came back as []. Downstream, score_headlines() then scored an
    empty list as a flat 50/Neutral, which is indistinguishable in the
    report from a real neutral sentiment read. "available" lets callers
    (and the report) show "Data Unavailable" instead of a fabricated
    Neutral when nothing could actually be fetched.
    """
    headlines = []
    sources_failed = []

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
        elif "status" in r and r.get("status") != "ok":
            # NewsAPI returned a well-formed error body (bad key, rate
            # limit, etc.) rather than raising -- treat that the same as
            # a failed fetch instead of silently moving on.
            sources_failed.append("NewsAPI")
            print(f"NewsAPI returned an error for {stock}: {r.get('message', r)}")
    except Exception as exc:
        sources_failed.append("NewsAPI")
        print(f"NewsAPI fetch failed for {stock}: {exc}")

    # Google RSS
    try:
        rss_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(stock)}"
        feed = feedparser.parse(rss_url)
        if getattr(feed, "bozo", False) and not feed.entries:
            # feedparser sets bozo=1 on a malformed/unreachable feed; if it
            # also came back with zero entries, treat this as a failure
            # rather than "no news" -- feedparser doesn't raise on its own.
            sources_failed.append("Google RSS")
            print(f"Google RSS fetch failed for {stock}: {getattr(feed, 'bozo_exception', 'malformed feed')}")
        else:
            for entry in feed.entries[:5]:
                headlines.append(entry.title)
    except Exception as exc:
        sources_failed.append("Google RSS")
        print(f"Google RSS fetch failed for {stock}: {exc}")

    # Only mark the whole fetch as unavailable if every source we tried
    # actually failed -- one source failing while the other genuinely
    # returns zero headlines is still a real (if empty) read.
    available = len(sources_failed) < 2

    return {
        "headlines": headlines,
        "available": available,
        "sources_failed": sources_failed,
    }