import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if you haven't already
nltk.download('vader_lexicon')

# Initialize VADER SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Function to extract names and calculate sentiment for "dressed" or "outfit" mentions
def analyze_best_worst_dressed(df):
    '''
    Analyzes sentiment of tweets mentioning "dressed" or "outfit" to identify best, worst, and controversial.
    
    Example output:
    {
        "Best Dressed": [{"Name": "Person 1", "Avg Sentiment": 0.85, "Number of Tweets": 15}, ...],
        "Worst Dressed": [{"Name": "Person 2", "Avg Sentiment": -0.65, "Number of Tweets": 10}, ...],
        "Most Controversial": [{"Name": "Person 3", "Sentiment Range": 1.5, "Number of Tweets": 8}, ...]
    }
    '''
    # Filter tweets with keywords
    df_filtered = df[df['clean_text'].str.contains(r'\b(dressed|outfit)\b', case=False)]
    
    # Extract mentions and calculate sentiment scores
    results = {}
    for index, row in df_filtered.iterrows():
        text = row['clean_text']
        sentiment = sid.polarity_scores(text)['compound']
        
        # Extract names
        name_patterns = [r'(\w+(?:\s+\w+)?)\s+(?:looked|was|is)\s+(?:stunning|gorgeous|beautiful|awful|terrible)',
                         r'(?:best|worst)\s+dressed\s+(\w+(?:\s+\w+)?)',
                         r'(\w+(?:\s+\w+)?)\s+in\s+(?:an\s+)?(?:outfit|dress)']
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for name in matches:
                if name not in results:
                    results[name] = []
                results[name].append(sentiment)
    
    # Classify into best, worst, and controversial based on sentiment
    best_dressed, worst_dressed, controversial_dressed = [], [], []
    for name, sentiments in results.items():
        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)
            sentiment_range = max(sentiments) - min(sentiments)
            num_tweets = len(sentiments)
            
            # Classify based on sentiment
            if avg_sentiment > 0.5:
                best_dressed.append({"Name": name, "Avg Sentiment": avg_sentiment, "Number of Tweets": num_tweets})
            elif avg_sentiment < -0.5:
                worst_dressed.append({"Name": name, "Avg Sentiment": avg_sentiment, "Number of Tweets": num_tweets})
            
            # Controversial classification based on sentiment range
            if sentiment_range > 1.0:
                controversial_dressed.append({"Name": name, "Sentiment Range": sentiment_range, "Number of Tweets": num_tweets})

    # Sort results
    best_dressed = sorted(best_dressed, key=lambda x: x["Avg Sentiment"], reverse=True)
    worst_dressed = sorted(worst_dressed, key=lambda x: x["Avg Sentiment"])
    controversial_dressed = sorted(controversial_dressed, key=lambda x: x["Sentiment Range"], reverse=True)
    
    # Return final structured output
    output = {
        "Best Dressed": best_dressed,
        "Worst Dressed": worst_dressed,
        "Most Controversial": controversial_dressed
    }
    
    return output
