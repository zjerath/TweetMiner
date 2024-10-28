import re

def extract_best_dressed_mentions(text):
    style_patterns = [
        r'(\w+(?:\s+\w+)?)\s+is\s+(?:one\s+of\s+the\s+)?best\s+dressed',
        r'(\w+(?:\s+\w+)?)\s+looked\s+(?:absolutely\s+)?(?:stunning|gorgeous|elegant|amazing)',
        r'best\s+dressed\s+goes\s+to\s+(\w+(?:\s+\w+)?)',
        r'(\w+(?:\s+\w+)?)\s+in\s+(?:a\s+)?(?:stunning|gorgeous|elegant|beautiful)\s+outfit'
    ]
    
    best_dressed = []
    for pattern in style_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        best_dressed.extend(matches)
    return best_dressed

def extract_all_best_dressed(df):
    df['best_dressed_mentions'] = df['clean_text'].apply(extract_best_dressed_mentions)
    
    all_mentions = df['best_dressed_mentions'].dropna()
    best_dressed_counts = {}

    for mentions in all_mentions:
        for name in mentions:
            best_dressed_counts[name] = best_dressed_counts.get(name, 0) + 1

    output = {
        "Best Dressed": [
            {"Name": name, "Number of Tweets": count} 
            for name, count in sorted(best_dressed_counts.items(), key=lambda item: item[1], reverse=True)
        ]
    }
    
    return output