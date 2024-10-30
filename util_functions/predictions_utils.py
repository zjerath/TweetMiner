import re
import spacy
nlp = spacy.load("en_core_web_lg")

def remove_punctuation(text):
    return ''.join(char for char in text if char.isalnum() or char.isspace())

# Function to apply regex patterns and extract potential nominees
def extract_potential_nominees(text, award):
    nominee_patterns = [
        r'(\w+(?:\s+\w+)?)\s+is\s+nominated\s+for\s+',
        r'(\w+(?:\s+\w+)?)\s+was\s+nominated\s+for\s+',
        r'(\w+(?:\s+\w+)?)\s+has\s+been\s+nominated\s+for\s+',
        r'nominee\s+(\w+(?:\s+\w+)?)\s+for\s+'
    ]
    
    def remove_punctuation(text):
            return ''.join(char for char in text if char.isalnum() or char.isspace())

    nominees = []
    # if remove_punctuation(award[:len(award)]).lower() in remove_punctuation(text).lower():
    if remove_punctuation(award).lower() in remove_punctuation(text).lower():
        for pattern in nominee_patterns:
            # filter tweets that contain the award name without punctuation
            matches = re.findall(pattern, text, re.IGNORECASE)
            nominees.extend(matches)
    return nominees


def extract_all_nominees(df, award):
    '''
    Returns a JSON with information about the award and a list of nominees based on the tweet data.
    Approach: 
    1. Extract potential nominees using regex patterns (x nominated for y). These nominees are weighted 3x.
    2. Apply NER to the tweets containing the award name. These nominees are weighted 1x. 
    
    Example output:
    {
        "Award": "Best Picture",
        "Nominees": [{"Name": "Nominee 1", "Number of Tweets": 10}, {"Name": "Nominee 2", "Number of Tweets": 5}, ...]
    }
    '''
    # Apply the extraction function to the 'clean_text' column
    df['potential_nominees'] = df['clean_text'].apply(lambda x: extract_potential_nominees(x, award))

    all_nominees = df['potential_nominees'].dropna()
    nominee_counts = {}

    for nominees in all_nominees:
        if nominees:  # Check if the list is not empty
            for nominee in nominees:
                if nominee in nominee_counts:
                    nominee_counts[nominee] += 3
                else:
                    nominee_counts[nominee] = 3

    # Filter tweets containing award name (without punctuation)
    clean_award = remove_punctuation(award).lower()
    filtered_df = df[df['clean_text'].apply(lambda x: clean_award in remove_punctuation(x).lower())]

    # Remove RT @ mentions and award name from tweets. These entities are picked up by NER. 
    filtered_df = filtered_df.copy()  
    filtered_df.loc[:, 'clean_text'] = filtered_df['clean_text'].str.replace('RT @\w+', '', regex=True)
    filtered_df.loc[:, 'clean_text'] = filtered_df['clean_text'].str.replace(clean_award, '', regex=False)
    filtered_df.loc[:, 'clean_text'] = filtered_df['clean_text'].str.replace(award, '', regex=False, case=False)
    
    # Apply NER to filtered tweets
    for _, row in filtered_df.iterrows():
        doc = nlp(row['clean_text'])
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'WORK_OF_ART']:
                name = ent.text
                if name not in nominee_counts:
                    nominee_counts[name] = 0
                nominee_counts[name] += 1

    # Create the JSON structure
    output = {
        "Award": award,
        "Nominees": [
            {
                "Name": nominee,
                "Number of Tweets": count
            } for nominee, count in nominee_counts.items()
        ]
    }

    # Sort the nominees by number of tweets in descending order
    output["Nominees"] = sorted(output["Nominees"], key=lambda x: x["Number of Tweets"], reverse=True)

    return output

# Function to apply regex patterns and extract potential winners
def extract_potential_winners(text, award):
    # Improved regex to properly handle 'just' variations
    just_variations = r'(?:(?:(?:she|he)\s+)?just\s+)?'
    winner_patterns = [
        r'(\w+(?:\s+\w+)?)\s+' + just_variations + r'wins\s+(?!' + award + ')',
        r'(\w+(?:\s+\w+)?)\s+' + just_variations + r'won\s+(?!' + award + ')',
        r'(\w+(?:\s+\w+)?)\s+' + just_variations + r'awarded\s+(?!' + award + ')',
        r'(\w+(?:\s+\w+)?)\s+' + just_variations + r'awarded\s+to\s+(?!' + award + ')',
        r'(\w+(?:\s+\w+)?)\s+' + just_variations + r'goes\s+(?!' + award + ')',
        r'(\w+(?:\s+\w+)?)\s+' + just_variations + r'received\s+(?!' + award + ')',
        # Regex pattern to capture "award - winner -" format
        r'(\w+(?:\s+\w+)?)\s+-\s+' + re.escape(award) + r'\s+-',
    ]
    winners = []
    for pattern in winner_patterns:
        # Remove stop words from pattern
        stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it',
                     'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
                     'have', 'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how'}
        
        pattern_words = pattern.split()
        pattern = ' '.join([word for word in pattern_words if word.lower() not in stop_words])
        matches = re.findall(pattern, text, re.IGNORECASE)
        winners.extend(matches)
    return winners

def extract_winners(df, award, nominees):
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
                if winner in nominees:
                    if winner in winner_counts:
                        winner_counts[winner] += 1
                    else:
                        winner_counts[winner] = 1

    # Create the JSON structure
    output = {
        "Award": award,
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

def extract_all_winners(df, awards, nominees):
    all_winners = []
    for award, nominee in zip(awards, nominees):
        all_winners.append(extract_winners(df, award, nominee))
    return all_winners

def extract_all_hosts(df):
    tweets = df[df['cleaned_text'].str.lower().str.contains('host')]['cleaned_text']

    return tweets.tolist()

def extract_all_presenters(df, award):
    tweets = df[df['cleaned_text'].str.lower().str.contains('present')]['cleaned_text']

    # filter tweets that contain the award name without punctuation
    tweets = tweets[tweets.apply(lambda x: remove_punctuation(award[:len(award)]).lower() in remove_punctuation(x).lower())]

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
