import spacy
import random
import json
import nltk
from nltk.metrics.distance import edit_distance
from util_functions.movie_data_utils import create_cast_crew_df



'''
Extract relevant PERSON & MOVIE entities from movies/credits data
'''
def define_entities(year):
    crew_df, cast_df = create_cast_crew_df(year)
    
    # combine distinct names into list - one for movies, one for people
    titles = cast_df['title'].unique()
    cast_names = list(cast_df['name'].unique())
    crew_names = list(crew_df['name'].unique())

    # ENTITY LISTS
    movie_entities = list(titles)
    people_entities = set(cast_names + crew_names)

    return movie_entities, people_entities


def named_entity_recognition(input):
    '''
    Extracts entities from the input text using spacy.

    Example Input: 
    [
        "Anne Hathaway is a great person", 
        "Jessica Chastain is also great", 
        ...
    ]

    Example Output: 
    [
        {
            'Entity': 'Anne Hathaway', 
            'Frequency': 2
        },
        ...
    ]
    '''
    spacy_model = spacy.load('en_core_web_lg')  # better entity recognition capability than en_core_web_sm

    entity_frequency = {}

    for text in input:
        doc = spacy_model(text)
        for entity in doc.ents:
            if entity.label_ == 'PERSON':
                entity_frequency[entity.text] = entity_frequency.get(entity.text, 0) + 1

    entity_list = [
        {
            'Name': entity,
            'Number of Tweets': frequency
        }
        for entity, frequency in entity_frequency.items()
    ]

    return sorted(entity_list, key=lambda x: x['Number of Tweets'], reverse=True)

def compute_edit_distance(string, entity_list):
    '''
    For a given string, compute edit distances against all possible entities
    Returns most similar matches from defined entity list
    '''
    entity_similarity_dict = {} # entity : similarity_score

    for entity in entity_list:
        # print(entity)
        try:
            similarity = edit_distance(string.lower(), entity.lower(), transpositions=True)
            # print(f"Entity: {entity} | Similarity: {similarity}")
            entity_similarity_dict[entity] = similarity
        except:
            pass

    return sorted( ((v,k) for k,v in entity_similarity_dict.items())) 

def token_overlap(query_string, classes):
    """
    Computes the most "likely" class for the given query string.

    First normalises the query to lower case, then computes the number of
    overlapping tokens for each of the possible classes.

    The class(es) with the highest overlap are returned as a list.

    """
    query_tokens = query_string.lower().split() # lowercase query
    class_tokens = [[x.lower() for x in c.split()] for c in classes] # lowercase each class in CLASSES
    # print(f"tokens:{class_tokens}")


    overlap = [0] * len(classes) # num times each word in query string appears for each defined CLASS 
    # check overlap on word/token level, not char
    for token in query_tokens:
        for index in range(len(classes)): 
            if token in class_tokens[index]:
                overlap[index] += 1

    # print(overlap)

    sorted_overlap = [(count, index) for index, count in enumerate(overlap)]
    sorted_overlap.sort()
    sorted_overlap.reverse()

    best_count = sorted_overlap[0][0]

    best_classes = []
    for count, index in sorted_overlap:
        if count == best_count and count > 0: # count > 0 -> DON'T FORCE MAPPING IF NO OVERLAP WITH ANY ENTITY
            best_classes.append(classes[index]) # (classes[index], count) to get token overlap count
        else:
            break

    return best_classes

"""
Given potential winners for an award, extract the top N candidates and winner
"""
def aggregate_candidates(potential_winners, entity_list, top_n=5):
    # data structure -> entity : count
    entity_count = {} 

    # LIMITING SEARCH TO TOP 50 CANDIDATES
    winners = sorted(potential_winners["Winners"][:20], key=lambda x: x["Number of Tweets"], reverse=True)

    # traverse names in winners
    for i in range(len(winners)):
        winner_info = winners[i] # name & tweet count
        winner_name = winner_info["Name"]
        winner_count = winner_info["Number of Tweets"]
        
        # identify entities "closest" to winner_name - quantified via similarity metric
        best_matches = compute_edit_distance(winner_name, entity_list=entity_list)[:top_n]
        best_match = best_matches[0][1] # [0] for top match, [1] for name
        
        # map name to entity, update entity count
        if best_match in entity_count:
            entity_count[best_match] += winner_count
        else:
            entity_count[best_match] = winner_count  

    # winner = entity w/ highest count
    candidate_dict = dict(sorted(entity_count.items(), key=lambda item: item[1], reverse=True))
    
    nominees = list(candidate_dict.keys())[:top_n]
    winner = nominees[0]
    nominees.remove(winner)
    
    return nominees, winner


def aggregate_entities(candidates, entity_list):
    '''
    Aggregates entities from the winners of a given award if some entities are named differently.
    For example, "Anne Hathaway" and "Anne Hathaway (actress)" would be considered the same entity.

    Example Input: 
    [
        {
            "Name": winner,
            "Number of Tweets": count
        },
        ...
    ]

    Example Output: 
    {
        "Anne Hathaway" : 10,
        "Ben Affleck" : 5,
        "Adele" : 3,
    }
    '''

    # data structure -> entity : count
    entity_count = {} 

    # LIMITING SEARCH TO TOP 50 CANDIDATES
    significant_candidates = candidates[:50]

    # traverse names in winners
    for i in range(len(significant_candidates)):
        candidate_info = significant_candidates[i] # name & tweet count
        candidate_name = candidate_info["Name"]
        candidate_count = candidate_info["Number of Tweets"]
        
        # identify entities "closest" to winner_name
        best_matches = compute_edit_distance(candidate_name, entity_list=entity_list) 
        best_match = best_matches[0][1]
            
        # map name to entity, update entity count
        if best_match in entity_count:
            entity_count[best_match] += candidate_count
        else:
            entity_count[best_match] = candidate_count  

    # winner = entity w/ highest count
    return dict(sorted(entity_count.items(), key=lambda item: item[1], reverse=True))


def format_human_readable(input):
    '''
    Formats a JSON input into a human-readable format. 

    Example Input:
    {
        "Event": "Golden Globes 2013",
        "Host": "Tina Fey",
        "Awards": [
            {   
                "Award": "Best Picture",
                "Nominees": [], 
                "Presenters": [], 
                "Winners": [
                    {
                        "Name": winner,
                        "Number of Tweets": count
                    },
                    ...
                ]
            }, 
            ...
        ]
    }

    Example Output:
    Event: Golden Globes 2013
    Host: Tina Fey

    Award: Best Picture
    Presenters: Presenter 1, Presenter 2, Presenter 3
    Nominees: Nominee 1, Nominee 2, Nominee 3, Nominee 4, Nominee 5
    Winner: Winner 1
    ...

    '''
    result = f"Event: {input['Event']}\n"
    result += f"Host: {input['Host']}\n\n"  

    for award in input['Awards']:
        result += f"Award: {award['Award']}\n"
        result += f"Presenters: {', '.join(award['Presenters'])}\n"
        result += f'''Nominees: {', '.join([f'"{nominee}"' for nominee in award['Nominees']])}\n'''
        result += f"Winner: \"{award['Winner']['Name']}\"\n\n"

    return result

def format_json(host, awards):
    json_result = {
        "Host": host
    }
    json_result.update(awards)
    return json.dumps(json_result, indent=4)