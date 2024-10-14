import numpy as np
import pandas as pd
import re
from ftfy import fix_text
import unidecode
import json

def preprocess_text(text):
    # Fix encoding issues (ampersands, etc.) using ftfy
    text = fix_text(text)
    
    # Remove non-ASCII characters (emojis, unicode symbols) using unidecode
    text = unidecode.unidecode(text)
    
    # Remove extra whitespace, tabs, and newlines (substitute with single spaces)
    # If we want to keep tabs/newline characters: text = re.sub(' +', ' ', text)
    text = " ".join(text.split())
    
    return text

def extract_hashtags_and_links(text):
    # Extract hashtags and links
    hashtags = re.findall(r'#\w+', text)  # Extract hashtags
    links = re.findall(r'http[s]?://\S+', text)  # Extract URLs
    
    # Remove hashtags and links from the original text
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs
    
    # Remove extra whitespace, tabs, and newlines (substitute with single spaces)
    # If we want to keep tabs/newline characters: text = re.sub(' +', ' ', text)
    text = " ".join(text.split())
    
    return text, hashtags, links

def preprocess_tweets(filename):
    df = pd.read_json(filename)

    # Extract user information into separate columns
    df['user_screen_name'] = df['user'].apply(lambda x: x['screen_name'])
    df['user_id'] = df['user'].apply(lambda x: x['id'])

    # Drop the original 'user' column as we've extracted the needed information
    df = df.drop('user', axis=1)

    # Convert timestamp_ms to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')

    # Drop the original timestamp_ms column
    df = df.drop('timestamp_ms', axis=1)

    # Reorder columns for better readability
    df = df[['id', 'timestamp', 'user_id', 'user_screen_name', 'text']]

    # Apply preprocessing
    df['clean_text'] = df['text'].apply(preprocess_text)

    # Display new text
    df['clean_text'].head()

    # Apply preprocessing
    df[['cleaned_text', 'hashtags', 'links']] = df['text'].apply(
        lambda x: pd.Series(extract_hashtags_and_links(x))
    )

    return df
