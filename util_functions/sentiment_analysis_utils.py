import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import spacy

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
    nlp = spacy.load("en_core_web_lg")
    # Filter tweets with keywords
    df_filtered = df[df['clean_text'].str.contains('dressed|outfit', case=False, regex=True)]
    
    # Remove duplicates by keeping first occurrence
    df_filtered = df_filtered.drop_duplicates(subset=['clean_text'], keep='first')
    
    # Filter out retweets
    df_filtered = df_filtered[~df_filtered['clean_text'].str.startswith('RT', na=False)]
    
    results = {}
    
    # Process each tweet
    for index, row in df_filtered.iterrows():
        print(f"Processing tweet {index}", row['clean_text'])
        text = row['clean_text']
        sentiment = sid.polarity_scores(text)['compound']
        
        # Use spaCy to extract person names
        doc = nlp(text)
        print(sentiment, doc.ents)
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                # Only store names that appear in context of dress/outfit
                name = ent.text
                context_window = 10  # Words before/after name to check for dress terms
                start_idx = max(0, ent.start - context_window)
                end_idx = min(len(doc), ent.end + context_window)
                context = doc[start_idx:end_idx].text.lower()
                
                if any(term in context for term in ['dress', 'outfit', 'wearing', 'looked']):
                    if name not in results:
                        results[name] = []
                    results[name].append(sentiment)

    best_dressed, worst_dressed, controversial_dressed = [], [], []
    print("Results dictionary:")
    print(f"Number of entries: {len(results)}")
    if len(results) == 0:
        print("Results dictionary is empty")
    else:
        print("Results dictionary contains data")
    print(results.items())
    for name, sentiments in results.items():
        print(f"Name: {name}, Sentiments: {sentiments}")
        # apply spacy to make sure name is a person
        doc = nlp(name)
        if len(doc.ents) == 0 or doc.ents[0].label_ != 'PERSON':
            continue
        
        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)
            sentiment_range = max(sentiments) - min(sentiments)
            num_tweets = len(sentiments)
            
            # Classify based on sentiment
            if avg_sentiment > 0.5 and num_tweets > 10:
                best_dressed.append({"Name": name, "Avg Sentiment": avg_sentiment, "Number of Tweets": num_tweets})
            elif avg_sentiment < .2 and num_tweets > 3:
                worst_dressed.append({"Name": name, "Avg Sentiment": avg_sentiment, "Number of Tweets": num_tweets})
            
            # Controversial classification based on sentiment range
            if sentiment_range > 1.0 and num_tweets > 5:
                controversial_dressed.append({"Name": name, "Sentiment Range": sentiment_range, "Number of Tweets": num_tweets})

    # Sort results
    best_dressed = sorted(best_dressed, key=lambda x: x["Avg Sentiment"] + x["Number of Tweets"]/100, reverse=True)
    worst_dressed = sorted(worst_dressed, key=lambda x: x["Avg Sentiment"] + x["Number of Tweets"]/100)
    controversial_dressed = sorted(controversial_dressed, key=lambda x: x["Sentiment Range"] + x["Number of Tweets"]/100, reverse=True)
    
    # Return final structured output
    output = {
        "Best Dressed": best_dressed,
        "Worst Dressed": worst_dressed,
        "Most Controversial": controversial_dressed
    }
    
    return output
