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