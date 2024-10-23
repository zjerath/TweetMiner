import pandas as pd
import ast
import os

'''
for each movie, retrieve data for associated cast members & crew members
Notable info:
- Cast: order of appearance in film
- Crew: job role (ex: Director, Writer, Producer)
intended to be used for checking eligible candidates for a given award
i.e. nominees for an award should be one of the "first N" actor/actresses to appear in a film
'''   
def create_cast_crew_df(year):    
    movies_credits_df = create_movies_credits_df(year)
    
    # Schema: title, character, gender, name, order (of appearance)
    cast_df = movies_credits_df[['title', 'cast']]
    cast_df = cast_df.explode('cast')
    cast_df['cast'].apply(pd.Series)
    cast_df = pd.concat([cast_df, cast_df['cast'].apply(pd.Series)], axis=1).drop('cast', axis=1)

    # Schema: title, job, name
    crew_df = movies_credits_df[['title', 'crew']]
    crew_df = crew_df.explode('crew')
    crew_df['crew'].apply(pd.Series)
    crew_df = pd.concat([crew_df, crew_df['crew'].apply(pd.Series)], axis=1).drop('crew', axis=1)

    return cast_df, crew_df

# create combined df w/ movies & credits
def create_movies_credits_df(year):
    # declare file paths
    movies_metadata_path = os.path.join(os.path.dirname(os.path.abspath('')), 'TweetMiner', 'data', 'movies_metadata.csv')
    credits_path = os.path.join(os.path.dirname(os.path.abspath('')), 'TweetMiner', 'data', 'credits.csv')

    movies = pd.read_csv(movies_metadata_path)
    credits = pd.read_csv(credits_path)

    # filter to 'Released' movies only
    movies = movies[movies['status']=='Released']
    
    # remove unnecessary columns
    movies.drop(columns=['belongs_to_collection', 'budget', 'homepage', 'imdb_id', 'overview', 'poster_path', 'runtime', 'status', 'tagline', 'video'], inplace=True)
    # function to check int id types
    def is_integer(val):
        try:
            # try to convert to int
            int(val)
            return True
        except (ValueError, TypeError):
            return False

    # filter rows where 'id' is an integer-like value
    movies = movies[movies.id.apply(is_integer)]

    # convert 'id' column to int
    movies.id = movies.id.astype(int)

    # merge with credits df
    df = pd.merge(movies, credits, on='id')
    df.drop(columns=['id'], inplace=True)

    # clean columns
    cols = ['genres', 'production_companies', 'production_countries', 'spoken_languages']
    for col in cols:
        df[col] = df[col].apply(extract_names)
    df.release_date = pd.to_datetime(df.release_date)
    df.cast = df.cast.apply(clean_cast_data)
    df.crew = df.crew.apply(clean_crew_data)

    # filter movie/credit data for relevant year
    df = df[df['release_date'].dt.year == year]

    return df

# extract the category names
def extract_names(name_str):
    if pd.isna(name_str):
        return []
    # convert the string representation of the list to an actual list
    str_list = ast.literal_eval(name_str)
    # extract the 'name' from each dictionary in the list
    names = [i['name'] for i in str_list]
    # return list of names as a string
    return ', '.join(names)

# clean the cast data
def clean_cast_data(cast_str):
    # convert string representation of the list to an actual list
    cast_list = ast.literal_eval(cast_str)

    # extract relevant fields and change gender values
    cleaned_cast = []
    for member in cast_list:
        cleaned_member = {
            'character': member['character'],
            'gender': 'm' if member['gender'] == 2 else 'f' if member['gender'] == 1 else None,
            'name': member['name'],
            'order': member['order']
        }
        cleaned_cast.append(cleaned_member)
    return cleaned_cast

# clean the crew data
def clean_crew_data(crew_str):
    # convert string representation of the list to an actual list
    crew_list = ast.literal_eval(crew_str)

    # extract relevant fields
    cleaned_crew = []
    for member in crew_list:
        cleaned_member = {
            'job': member['job'],
            'name': member['name']
        }
        cleaned_crew.append(cleaned_member)
    return cleaned_crew