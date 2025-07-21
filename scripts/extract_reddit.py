import praw
import pandas as pd
import time
import os
from datetime import datetime
from dotenv import load_dotenv


# 🔹 Step 1: Reddit API Credentials (Replace with your values)
load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent=os.getenv("USER_AGENT")
)


# 🔹 Step 2: Define Keywords & Subreddits
search_terms = [
    "HajjExperience", "Hajj", "HajjJourney", "HajjAndUmrah",
    "hajj pilgrimage", "Makkah", "Kabba", "Madina"
]
subreddits = ["hajj", "islam", "travel", "muslim"]

# 🔹 Step 3: Fetch Reddit Posts
posts_data = []
for subreddit in subreddits:
    for term in search_terms:
        for post in reddit.subreddit(subreddit).search(term, limit=100):
            posts_data.append({
                "post_id": post.id,
                "subreddit": subreddit,
                "title": post.title,
                "text": post.selftext,
                "created_at": post.created_utc,
                "url": post.url
            })

# 🔹 Step 4: Convert to DataFrame & Save
df = pd.DataFrame(posts_data)
df.to_csv("data/reddit_hajj_data.csv", index=False)

print("✅ Data saved! Check 'reddit_hajj_data.csv'.")
