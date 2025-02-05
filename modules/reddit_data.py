import praw
import pandas as pd
import time

reddit = praw.Reddit(
    client_id="fGDuCosBvBG49tfpZYg2Kw",
    client_secret="",
    user_agent="windows:Reapes:v1.0 (by /u/Infamous-Version2359)"
)

DATA_FILE = "data/reddit_data.csv"

def fetch_reddit_data(keyword, post_limit=100, max_comments=50, max_runtime=300):
    data = []
    start_time = time.time()

    for post in reddit.subreddit("all").search(keyword, limit=post_limit):
        if time.time() - start_time > max_runtime:
            break  

        try:
            post.comments.replace_more(limit=0)
            comments = [{
                "comment_body": c.body,
                "comment_author": str(c.author),
                "comment_score": c.score
            } for c in post.comments.list()[:max_comments] if c.body]

            data.append({
                "subreddit": post.subreddit.display_name,
                "post_title": post.title or "No Title",
                "post_content": post.selftext or "No Content",
                "post_author": str(post.author),
                "post_score": post.score,
                "post_url": post.url,
                "post_created_utc": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(post.created_utc)),
                "comments": comments
            })

        except:
            continue

    return data

def save_data_to_csv(data):
    if not data:
        return

    rows = []
    for post in data:
        for comment in post["comments"]:
            rows.append({
                "subreddit": post["subreddit"],
                "post_title": post["post_title"],
                "post_content": post["post_content"],
                "post_author": post["post_author"],
                "post_score": post["post_score"],
                "post_url": post["post_url"],
                "post_created_utc": post["post_created_utc"],
                "comment_body": comment["comment_body"],
                "comment_author": comment["comment_author"],
                "comment_score": comment["comment_score"]
            })

    if rows:
        pd.DataFrame(rows).to_csv(DATA_FILE, index=False, encoding="utf-8")
