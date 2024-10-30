import re

def extract_best_dressed_mentions(text):
    style_patterns = [
        r'(\w+(?:\s+\w+)?)\s+is\s+(?:one\s+of\s+the\s+)?best\s+dressed',
        r'(\w+(?:\s+\w+)?)\s+looked\s+(?:absolutely\s+)?(?:stunning|gorgeous|elegant|amazing)',
        r'best\s+dressed\s+goes\s+to\s+(\w+(?:\s+\w+)?)',
        r'(\w+(?:\s+\w+)?)\s+in\s+(?:a\s+)?(?:stunning|gorgeous|elegant|beautiful)\s+outfit',
        r'(\w+(?:\s+\w+)?)\s+worst\s+dressed',
        r'(\w+(?:\s+\w+)?)\s+looked\s+(?:absolutely\s+)?(?:terrible|awful|bad|horrible)',
        r'worst\s+dressed\s+goes\s+to\s+(\w+(?:\s+\w+)?)',
        r'(\w+(?:\s+\w+)?)\s+in\s+(?:a\s+)?(?:terrible|awful|ugly|hideous)\s+(?:dress|outfit|gown)',
        r'(\w+(?:\s+\w+)?)\s+wearing\s+(?:a\s+)?(?:beautiful|stunning|gorgeous)\s+(?:dress|gown)',
        r'(\w+(?:\s+\w+)?)\s+(?:dress|outfit|gown)\s+is\s+(?:beautiful|stunning|gorgeous)',
        r'(\w+(?:\s+\w+)?)\s+fashion\s+(?:win|fail)',
        r'(\w+(?:\s+\w+)?)\s+(?:nailed|failed)\s+(?:it|the\s+look)'
    ]
    
    best_dressed = []
    for pattern in style_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        best_dressed.extend(matches)
    return best_dressed

def extract_all_best_dressed(df):
    df['best_dressed_mentions'] = df['clean_text'].apply(extract_best_dressed_mentions)

    # Remove duplicates within each tweet's mentions
    df['best_dressed_mentions'] = df['best_dressed_mentions'].apply(lambda x: list(dict.fromkeys(x)) if isinstance(x, list) else x)
    
    # Filter out retweets
    df = df[~df['clean_text'].str.startswith('RT', na=False)]
    
    all_mentions = df['best_dressed_mentions'].dropna()
    best_dressed_counts = {}

    for mentions in all_mentions:
        # use spacy to extract entities
        doc = nlp(mentions)
        for ent in doc.ents:
            best_dressed_counts[ent.text] = best_dressed_counts.get(ent.text, 0) + 1

    output = {
        "Best Dressed": [
            {"Name": name, "Number of Tweets": count} 
            for name, count in sorted(best_dressed_counts.items(), key=lambda item: item[1], reverse=True)
        ]
    }
    
    return output