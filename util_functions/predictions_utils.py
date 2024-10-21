import re

# Function to apply regex patterns and extract potential nominees
def extract_potential_nominees(text, award):
    nominee_patterns = [
        r'(\w+(?:\s+\w+)?)\s+is\s+nominated\s+for\s+' + award,
        r'(\w+(?:\s+\w+)?)\s+was\s+nominated\s+for\s+' + award,
        r'(\w+(?:\s+\w+)?)\s+has\s+been\s+nominated\s+for\s+' + award,
        r'nominee\s+(\w+(?:\s+\w+)?)\s+for\s+' + award,
        r'nominated\s+for\s+' + award + r'\s+(\w+(?:\s+\w+)?)'
    ]
    
    nominees = []
    for pattern in nominee_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        nominees.extend(matches)
    return nominees


def extract_all_nominees(df, award, presenters=[]):
    '''
    Returns a JSON with information about the award and a list of nominees based on the tweet data.
    
    Example output:
    {
        "Award": "Best Picture",
        "Nominees": ["Nominee 1", "Nominee 2", "Nominee 3", "Nominee 4", "Nominee 5"], 
        "Presenters": ["Presenter 1", "Presenter 2", "Presenter 3"],
    }
    '''
    # Apply the extraction function to the 'clean_text' column
    df['potential_nominees'] = df['clean_text'].apply(lambda x: extract_potential_nominees(x, award))

    all_nominees = df['potential_nominees'].dropna()
    nominee_list = []

    for nominees in all_nominees:
        if nominees:  # Check if the list is not empty
            nominee_list.extend(nominees)

    # Remove duplicates by converting to a set and then back to a list
    nominee_list = list(set(nominee_list))

    # Create the JSON structure
    output = {
        "Award": award,
        "Nominees": nominee_list,
        "Presenters": presenters,  # We don't have presenter information in the current data
    }

    return output

# Function to apply regex patterns and extract potential winners
def extract_potential_winners(text, award):
    # Improved regex to properly handle 'just' variations
    just_variations = r'(?:(?:(?:she|he)\s+)?just\s+)?'
    winner_patterns = [
        r'(\w+(?:\s+\w+)?)\s+' + just_variations + r'wins\s+(?!' + award + ')',
        r'(\w+(?:\s+\w+)?)\s+' + just_variations + r'won\s+(?!' + award + ')',
        r'(\w+(?:\s+\w+)?)\s+' + just_variations + r'awarded\s+(?!' + award + ')',
        r'(\w+(?:\s+\w+)?)\s+' + just_variations + r'receives\s+(?!' + award + ')',
        r'(\w+(?:\s+\w+)?)\s+' + just_variations + r'received\s+(?!' + award + ')'
    ]
    winners = []
    for pattern in winner_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        winners.extend(matches)
    return winners

def extract_all_winners(df, award, nominees=[], presenters=[]):
    '''
    Returns a JSON with the information about the award, and a list of winners and the number of tweets they were mentioned in as a winner. 

    Example output: 
    {
        "Award": "Best Picture",
        "Nominees": ["Nominee 1", "Nominee 2", "Nominee 3", "Nominee 4", "Nominee 5"], 
        "Presenters": ["Presenter 1", "Presenter 2", "Presenter 3"],
        "Winners": [
            {
                "Name": Winner 1,
                "Number of Tweets": 512
            },
            {
                "Name": Winner 2,
                "Number of Tweets": 123
            },
            ...
        ]
    }
    '''
    # Apply the extraction function to the 'text' column
    df['potential_winners'] = df['clean_text'].apply(lambda x: extract_potential_winners(x, award))

    # Print all non-NaN values in the potential_winners column
    all_winners = df['potential_winners'].dropna()
    winner_counts = {}
    for winners in all_winners:
        if winners:  # Check if the list is not empty
            for winner in winners:
                if winner in winner_counts:
                    winner_counts[winner] += 1
                else:
                    winner_counts[winner] = 1

    # Create the JSON structure
    output = {
        "Award": award,
        "Nominees": nominees,  # We don't have nominee information in the current data
        "Presenters": presenters,  # We don't have presenter information in the current data
        "Winners": [
            {
                "Name": winner,
                "Number of Tweets": count
            } for winner, count in winner_counts.items()
        ]
    }

    # Sort the winners by number of tweets in descending order
    output["Winners"] = sorted(output["Winners"], key=lambda x: x["Number of Tweets"], reverse=True)

    return output

def extract_all_hosts(df):
    tweets = df[df['cleaned_text'].str.lower().str.contains('host')]['cleaned_text']

    return tweets.tolist()

def extract_all_award_names(df):
    tweets = df[df['cleaned_text'].str.contains('Best')]['cleaned_text']

    # Filter tweets that contain only one 'best'
    tweets = tweets[tweets.str.count('Best') == 1]
    
    # Extract the part of the tweet from 'best' to the end of the sentence or 'goes to', excluding punctuation
    tweets = tweets.apply(lambda x: re.search(r'best.*?(?=[.!?:]|goes to|win|won)', x, re.IGNORECASE))

    # Filter out tweets containing certain words
    blacklist_words = ['@', '&', 'golden globes', 'oscars', 'known for', 'speech', 'outfit', 'dress', 'look', 'carpet', 'interview', 'night', 
                       'joke', 'clip', 'celebration', 'so far', 'of all time', 'of the', 'at the','ever', 'fan', 'surpris', 'buy', 'award', 
                       'win', 'won', 'nominated', 'hotel']
    
    for word in blacklist_words:
        tweets = tweets[~tweets.apply(lambda x: word.lower() in x.group().lower() if x else False)]
    
    # Filter tweets and keep only the part before the second '-' if there are more than one
    def filter_dashes(tweet):
        if not tweet:
            return None
        parts = tweet.group().split('-')
        if len(parts) > 1:
            return f"{parts[0].strip()} - {parts[1].strip()}"
        return parts[0].strip()

    tweets = tweets.apply(filter_dashes)
    
    # Remove any trailing whitespace
    tweets = tweets.apply(lambda x: x.strip() if x else None)

    # Remove tweets of length 1
    tweets = tweets[tweets.apply(lambda x: len(x.split()) > 1 if x else False)]
    
    # Keep only the matched parts and convert to a list
    tweets = tweets.dropna().tolist()
    
    # Count occurrences of each award name
    award_counts = {}
    for tweet in tweets:
        tweet_lower = tweet.lower()
        if tweet_lower in award_counts:
            award_counts[tweet_lower]['count'] += 1
        else:
            award_counts[tweet_lower] = {'original': tweet, 'count': 1}
    
    # Convert the dictionary to preserve original capitalization
    award_counts = {v['original']: v['count'] for v in award_counts.values()}
    
    # Create list of dictionaries with award names and occurrences
    award_list = [
        {
            "Name": award,
            "Number of Tweets": count
        } for award, count in award_counts.items()
    ]
    
    # Sort the list by number of occurrences in descending order
    award_list = sorted(award_list, key=lambda x: x["Number of Tweets"], reverse=True)

    return award_list
