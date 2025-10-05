import os
import praw
from transformers import pipeline
from urllib.parse import urlparse
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


print(f"Loading .env from: {env_path}")
print(f"Environment file exists: {env_path.exists()}")

try:
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT"),
    )
    print("Reddit client initialized successfully")
except Exception as e:
    print(f"Error initializing Reddit client: {e}")
    exit(1)


print("Loading emotion analysis model...")
emotion_analyzer = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,  
    return_all_scores=True
)
print("Emotion analyzer loaded successfully")

def get_subreddit_name_from_url(url: str) -> str:
    """Extract subreddit name from URL like https://www.reddit.com/r/taylorswift/"""
    try:
        path_parts = urlparse(url).path.strip("/").split("/")
        if len(path_parts) >= 2 and path_parts[0] == "r":
            return path_parts[1]
        raise ValueError("Invalid subreddit URL format")
    except Exception as e:
        raise ValueError(f"Invalid subreddit URL. Example: https://www.reddit.com/r/taylorswift/. Error: {e}")

def analyze_text_emotion(text: str) -> dict:
    if not text.strip():
        return {"primary_emotion": "neutral", "confidence": 0.0, "all_emotions": {}}
    
    try:
        emotions = emotion_analyzer(text[:512])  # Limit text length for model
        
        emotion_scores = {emotion['label'].lower(): emotion['score'] for emotion in emotions[0]}
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        
        return {
            "primary_emotion": primary_emotion[0],
            "confidence": primary_emotion[1],
            "all_emotions": emotion_scores
        }
    except Exception as e:
        print(f"Error analyzing emotion: {e}")
        return {"primary_emotion": "unknown", "confidence": 0.0, "all_emotions": {}}

def analyze_subreddit(url: str, limit: int = 10, sort_by: str = "new", save_results: bool = True):
    try:
        subreddit_name = get_subreddit_name_from_url(url)
        subreddit = reddit.subreddit(subreddit_name)
        
        print(f"🔎 Fetching {limit} {sort_by} posts from r/{subreddit_name}...")
        print(f"📊 Analyzing emotions using DistilRoBERTa model\n")
        
        if sort_by == "new":
            posts = subreddit.new(limit=limit)
        elif sort_by == "hot":
            posts = subreddit.hot(limit=limit)
        elif sort_by == "top":
            posts = subreddit.top(limit=limit, time_filter="week")
        elif sort_by == "rising":
            posts = subreddit.rising(limit=limit)
        else:
            posts = subreddit.new(limit=limit)
        
        results = []
        emotion_counts = Counter()
        
        for i, post in enumerate(posts, 1):
            print(f"Analyzing post {i}/{limit}...")
            
            text_to_analyze = f"{post.title}. {post.selftext}" if post.selftext else post.title
            
            if not text_to_analyze.strip():
                continue
            
            emotion_result = analyze_text_emotion(text_to_analyze)
            
            emotion_counts[emotion_result["primary_emotion"]] += 1
            
            result = {
                "title": post.title,
                "url": f"https://www.reddit.com{post.permalink}",
                "author": str(post.author) if post.author else "[deleted]",
                "score": post.score,
                "num_comments": post.num_comments,
                "created_utc": datetime.fromtimestamp(post.created_utc),
                "primary_emotion": emotion_result["primary_emotion"],
                "confidence": emotion_result["confidence"],
                "text_length": len(text_to_analyze)
            }
            
            for emotion, score in emotion_result["all_emotions"].items():
                result[f"emotion_{emotion}"] = score
            
            results.append(result)
            
            print(f"Title: {post.title[:80]}{'...' if len(post.title) > 80 else ''}")
            print(f"Author: {result['author']} | 📊 Score: {post.score} | 💬 Comments: {post.num_comments}")
            print(f"Primary Emotion: {emotion_result['primary_emotion'].upper()} (confidence: {emotion_result['confidence']:.3f})")
            
            top_emotions = sorted(emotion_result["all_emotions"].items(), key=lambda x: x[1], reverse=True)[:3]
            emotion_str = " | ".join([f"{emotion}: {score:.3f}" for emotion, score in top_emotions])
            print(f"Top emotions: {emotion_str}")
            print(f"URL: {result['url']}\n")
        
        print("=" * 80)
        print(f"EMOTION ANALYSIS SUMMARY for r/{subreddit_name}")
        print("=" * 80)
        
        total_posts = len(results)
        if total_posts > 0:
            for emotion, count in emotion_counts.most_common():
                percentage = (count / total_posts) * 100
                print(f"🎭 {emotion.capitalize()}: {count} posts ({percentage:.1f}%)")
            
            avg_confidence = sum(r["confidence"] for r in results) / len(results)
            print(f"\nAverage confidence: {avg_confidence:.3f}")
            
            if save_results and results:
                df = pd.DataFrame(results)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"reddit_emotion_analysis_{subreddit_name}_{timestamp}.csv"
                df.to_csv(filename, index=False)
                print(f"Results saved to: {filename}")
                
                create_emotion_plot(emotion_counts, subreddit_name, timestamp)
        
        return results
        
    except Exception as e:
        print(f"Error analyzing subreddit: {e}")
        return []

def create_emotion_plot(emotion_counts: Counter, subreddit_name: str, timestamp: str):
    """Create and save emotion distribution plot"""
    try:
        plt.figure(figsize=(12, 8))
        
        emotions = list(emotion_counts.keys())
        counts = list(emotion_counts.values())
        colors = plt.cm.Set3(range(len(emotions)))
        
        bars = plt.bar(emotions, counts, color=colors)
        
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.title(f'Emotion Distribution in r/{subreddit_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Emotions', fontsize=12)
        plt.ylabel('Number of Posts', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        total = sum(counts)
        plt.text(0.02, 0.98, f'Total Posts: {total}', transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        plot_filename = f"emotion_distribution_{subreddit_name}_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Emotion distribution plot saved to: {plot_filename}")
        
    except Exception as e:
        print(f"Error creating plot: {e}")

def analyze_multiple_subreddits(subreddit_urls: list, limit: int = 10):
    all_results = {}
    
    for url in subreddit_urls:
        try:
            subreddit_name = get_subreddit_name_from_url(url)
            print(f"\n{'='*20} Analyzing r/{subreddit_name} {'='*20}")
            results = analyze_subreddit(url, limit=limit, save_results=False)
            all_results[subreddit_name] = results
        except Exception as e:
            print(f"Error analyzing {url}: {e}")
            continue
    
    if len(all_results) > 1:
        print(f"\n{'='*20} COMPARISON ACROSS SUBREDDITS {'='*20}")
        for subreddit, results in all_results.items():
            if results:
                emotion_counts = Counter(r["primary_emotion"] for r in results)
                most_common = emotion_counts.most_common(1)[0]
                avg_confidence = sum(r["confidence"] for r in results) / len(results)
                print(f"r/{subreddit}: Most common emotion is '{most_common[0]}' ({most_common[1]} posts, avg confidence: {avg_confidence:.3f})")

def interactive_analysis():
    print("Reddit Emotion Analyzer - Interactive Mode")
    print("=" * 50)
    
    while True:
        try:
            url = input("\nEnter subreddit URL (or 'quit' to exit): ").strip()
            if url.lower() in ['quit', 'exit', 'q']:
                break
            
            limit = input("Number of posts to analyze (default: 10): ").strip()
            limit = int(limit) if limit.isdigit() else 10
            
            sort_by = input("Sort by (new/hot/top/rising, default: new): ").strip().lower()
            if sort_by not in ['new', 'hot', 'top', 'rising']:
                sort_by = 'new'
            
            analyze_subreddit(url, limit=limit, sort_by=sort_by)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    client_id = os.getenv("REDDIT_CLIENT_ID")
    if not client_id:
        print("REDDIT_CLIENT_ID not found in environment variables")
        print("Make sure your .env file contains the required Reddit API credentials")
        exit(1)
    
    print("Reddit Emotion Analyzer Started")
    print("=" * 50)
    
    try:
        url = "https://www.reddit.com/r/taylorswift/"
        analyze_subreddit(url, limit=5, sort_by="hot")
        
    except KeyboardInterrupt:
        print("\n Analysis interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")