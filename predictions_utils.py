import re

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
    tweets = df[df['cleaned_text'].str.contains('hosting')]['cleaned_text']

    return tweets.tolist()


