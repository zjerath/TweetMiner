import spacy
import random
import json

ENTITIES = [
    'Ben Affleck', 
    'Anne Hathaway', 
    'Julianne Moore', 
    'Adele', 
    'Jessica Chastain', 
    'Daniel Day-Lewis', 
    'Denzel Washington', 
    'Jonah Hill', 
    'Brad Pitt', 
    'Amy Poehler'
]

def extract_entities(input):
    '''
    Extracts entities from the winners of a given award.

    Example Input: 
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
    }

    Example Output: 
    [
        {'Name': 'Ben Affleck', 'Entities': ['Ben Affleck']},
        {'Name': 'Anne Hathaway', 'Entities': ['Anne Hathaway']},
        {'Name': 'Hugh Jackman', 'Entities': ['Hugh Jackman']},
        {'Name': 'Jennifer Lawrence', 'Entities': ['Jennifer Lawrence']},
        {'Name': 'Adele', 'Entities': ['Adele']},
        {'Name': 'when she', 'Entities': []}
    ]
    '''
    winners = sorted(input["Winners"], key=lambda x: x["Number of Tweets"], reverse=True)

    spacy_model = spacy.load('en_core_web_lg') # better entity recognition capability than en_core_web_sm

    entity_list = []

    for i in range(len(winners)):
        winner_name = winners[i]["Name"]
        spacy_output = spacy_model(winner_name)
        # print(f"spacy output: {spacy_output.ents}")
        # if spacy_output.ents == (): print("NO ENTITY IDENTIFIED")
        associated_entities = []
        for entity in spacy_output.ents:
            # print(f"entity:{entity}")
            # print([entity.text, entity.label_])
            # entity_list.append(entity.text)
            associated_entities.append(entity.text)
        
        name_entities = {
            "Name" : winner_name,
            "Entities" : associated_entities
        }
        
        entity_list.append(name_entities)
        
    return entity_list

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

def aggregate_entities(input):
    '''
    Aggregates entities from the winners of a given award if some entities are named differently.
    For example, "Anne Hathaway" and "Anne Hathaway (actress)" would be considered the same entity.

    Example Input: 
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
    }

    Example Output: 
    {
        "Anne Hathaway" : 10,
        "Ben Affleck" : 5,
        "Adele" : 3,
    }
    '''

    # data structure -> entity : [different names]
    entity_names = {key: [] for key in ENTITIES}

    # data structure -> entity : count
    entity_count = {key: 0 for key in ENTITIES} 

    winners = sorted(input["Winners"], key=lambda x: x["Number of Tweets"], reverse=True)

    # traverse names in winners
    for i in range(len(winners)):
        winner_info = winners[i] # name & tweet count
        winner_name = winner_info["Name"]
        winner_count = winner_info["Number of Tweets"]

        # print(winner_info)
        
        # identify entities "closest" to winner_name - replace token_overlap w/ any similarity metric
        candidate_entities = token_overlap(winner_name, classes=ENTITIES) 
            
        # don't map if no entity recognized
        if len(candidate_entities) == 0: continue
        
        # typically single candidate identified, but in case multiple top candidates named pick random - should probably change
        identified_entity = random.choice(candidate_entities) 
        
        # print(f"Name: {winner_name} | Candidate entities: {candidate_entities} | Identified entity: {identified_entity}")

        # map name to entity, update entity count
        entity_names[identified_entity].append(winner_name)
        entity_count[identified_entity] += winner_count

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