import json
import pandas as pd
from util_functions.preprocessing_utils import preprocess_tweets
from util_functions.predictions_utils import extract_all_winners, extract_all_hosts, extract_all_award_names, extract_all_nominees, extract_all_presenters, extract_potential_nominees
from util_functions.aggregation_utils import aggregate_candidates, aggregate_entities, format_human_readable, named_entity_recognition, define_entities, is_person_name

def import_data():
    with open("data/gg2013answers.json", 'r') as f:
        answers_data = json.load(f)
    hosts = answers_data['hosts']
    award_data = answers_data['award_data']

    return hosts, award_data

def find_hosts(df):
    """
    Find hosts for the entire ceremony.
    """
    # Define entity list of people which host could come from
    # _, people_entities = define_entities(year)

    hosts_tweets = extract_all_hosts(df)
    hosts_entities = named_entity_recognition(hosts_tweets)

    hosts_entities = aggregate_entities(hosts_entities)

    # Calculate the mean and standard deviation of the counts
    counts = list(hosts_entities.values())
    mean = sum(counts) / len(counts)

    std_dev = (sum((x - mean) ** 2 for x in counts) / len(counts)) ** 0.5

    # Define a threshold as 1.5 standard deviations above the mean
    threshold = mean + 1.5 * std_dev

    significant_hosts = [(entity, count) for entity, count in hosts_entities.items() if count > threshold] 
    # If no hosts meet the threshold, return the most mentioned entity with its count 
    if not significant_hosts: 
        sorted_candidates = sorted(hosts_entities.items(), key=lambda x: x[1], reverse=True) 
        significant_hosts = [sorted_candidates[0]] # Keep the top entity with its count 
        return significant_hosts
        # Sort the significant hosts by count in descending order 
    significant_hosts.sort(key=lambda x: x[1], reverse=True) 
    return significant_hosts
    
    
    # # Get the hosts that are mentioned with frequency above the threshold
    # significant_hosts = [entity for entity, count in hosts_entities.items() if count > threshold]

    # # If no hosts meet the threshold, return the most mentioned entity
    # if not significant_hosts:
    #     sorted_candidates = sorted(hosts_entities.items(), key=lambda x: x[1], reverse=True)
    #     significant_hosts = [entity for entity, _ in sorted_candidates[:1]]
    
    # return significant_hosts

def find_nominees(df, awards, top_n):
    # takes in the preprocessed df and hard-coded list of awards
    top_nominees_by_award = []
    for award in awards:
        award_nominees = extract_all_nominees(df, award)
        # Get the top 5 nominees
        top_nominees = award_nominees["Nominees"][:top_n]
        # Store the result in the list
        top_nominees_by_award.append({
            "Award": award,
            f"Top {top_n} Nominees": top_nominees
        })
    return top_nominees_by_award
    
def find_award_names(df):
    
    award_names = extract_all_award_names(df)

    # Calculate the mean and standard deviation of the counts
    counts = [award['Number of Tweets'] for award in award_names]
    mean = sum(counts) / len(counts)
    std_dev = (sum((x - mean) ** 2 for x in counts) / len(counts)) ** 0.5

    # Define a threshold as 1 standard deviation above the mean
    threshold = mean + 1 * std_dev

    # Filter awards that are mentioned with frequency above the threshold
    significant_awards = [award for award in award_names if award['Number of Tweets'] > threshold]

    # Sort the significant awards by number of tweets in descending order
    significant_awards.sort(key=lambda x: x['Number of Tweets'], reverse=True)

    # Update award_names to only include significant awards
    award_names = significant_awards

    # TODO: If award contains a name, remove it from the list
    # TODO: Make sure award capitalization is correct. If not, remove it from the list

    return award_names

def get_award_winner(df, award_name, nominees):
    potential_award_winners = extract_all_winners(df, award=award_name, nominees=[], presenters=[])

    candidate_dict = dict(sorted(potential_award_winners.items(), key=lambda item: item[1], reverse=True))
    
    nominees = list(candidate_dict.keys())[:6]
    winner = nominees[0]
    nominees.remove(winner)

    # nominees, winner = aggregate_candidates(potential_winners=potential_award_winners, entity_list=people_entities)

    print(f"Nominees for {award_name}: {nominees}")
    print(f"Winner of {award_name}: {winner}")

def get_award_presenters(df, award_name, hosts):
    presenters = extract_all_presenters(df, award_name)
    presenters_entities = named_entity_recognition(presenters)

    # Filter out non-person names
    presenters_entities = [entity for entity in presenters_entities if is_person_name(entity['Name'])]

    # Filter out hosts
    presenters_entities = [presenter for presenter in presenters_entities if presenter['Name'] not in hosts]

    # Sort presenters by number of mentions in descending order and take the top 3
    presenters_entities = sorted(presenters_entities, key=lambda x: x['Number of Tweets'], reverse=True)[:3]

    return presenters_entities

def get_presenters(df, award_names, hosts):
    presenters = []
    for award in award_names:
        award_presenters = get_award_presenters(df, award, hosts)
        presenters.append({
            "Award Name": award,
            "Presenters": [presenter['Name'] for presenter in award_presenters]
        })

    return presenters


def main(year):
    # this is for hard-coded stuff to prevent cascading error
    with open(f"data/gg{year}answers.json", 'r') as f:
        answers_data = json.load(f)
    hardcoded_hosts_data = answers_data['hosts']
    hardcoded_awards_data = answers_data['award_data']
    hardcoded_award_names = list(hardcoded_awards_data.keys())
    

    df = preprocess_tweets(f"data/gg{year}.json")

    # print("NOMINEES...")
    # x = df['clean_text'].apply(lambda x: extract_potential_nominees(x, hardcoded_award_names[0]))
    # print(x)
    
    print("HOSTS...")
    hosts = find_hosts(df)
    print(hosts)

    print("AWARD NAMES...")
    awards = find_award_names(df)
    print(awards)

    '''
    # using our award & host predictions to retrieve presenters
    print("PRESENTERS...")
    presenters = get_presenters(df, awards, hosts)
    print(presenters)
    '''

    print("PRESENTERS...")
    presenters = get_presenters(df, hardcoded_award_names, hardcoded_hosts_data)
    print(presenters)

    '''
    # using our award & host predictions to retrieve nominees
    print("NOMINEES...")
    nominees = find_nominees(df, awards, 5)
    print(nominees)
    '''

    print("NOMINEES...")
    x = df['clean_text'].apply(lambda x: extract_potential_nominees(x, hardcoded_award_names[0]))
    print(x)
    nominees = find_nominees(df, hardcoded_award_names, 5)
    print(nominees)

    '''
    # using our award & nominee predictions to retrieve winners
    print("WINNERS...")
    winners = find_nominees(df, awards, 5)
    print(winners)
    '''

    # print("NOMINEES...")
    # nominees = find_nominees(df, hardcoded_award_names, 5)
    # print(nominees)

    # # awards list
    # awards = ["best screenplay - motion picture", "best director - motion picture", 
    #       "best performance by an actress in a television series - comedy or musical",
    #       "best foreign language film",
    #       "best performance by an actor in a supporting role in a motion picture",
    #       "best performance by an actress in a supporting role in a series, mini-series or motion picture made for television",
    #       "best motion picture - comedy or musical",
    #       "best performance by an actress in a motion picture - comedy or musical",
    #       "best mini-series or motion picture made for television",
    #       "best original score - motion picture",
    #       "best performance by an actress in a television series - drama",
    #       "best performance by an actress in a motion picture - drama",
    #       "cecil b. demille award",
    #       "best performance by an actor in a motion picture - comedy or musical",
    #       "best motion picture - drama",
    #       "best performance by an actor in a supporting role in a series, mini-series or motion picture made for television",
    #       "best performance by an actress in a supporting role in a motion picture",
    #       "best television series - drama",
    #       "best performance by an actor in a mini-series or motion picture made for television",
    #       "best performance by an actress in a mini-series or motion picture made for television",
    #       "best animated feature film",
    #       "best original song - motion picture",
    #       "best performance by an actor in a motion picture - drama",
    #       "best television series - comedy or musical",
    #       "best performance by an actor in a television series - drama",
    #       "best performance by an actor in a television series - comedy or musical"
    #       ]

    # award_winners_output = extract_all_winners(df, "Best Picture", nominees=[], presenters=[])

    # award_winners_output['Winners'] = aggregate_entities(award_winners_output['Winners'])

    # # Get the winner with the highest value
    # winner = max(award_winners_output['Winners'], key=award_winners_output['Winners'].get)

    # award_winners_output['Winner'] = winner
    
    # # Get the count of mentions for the winner
    # winner_count = award_winners_output['Winners'][winner]

    # # Remove the 'Winners' key from the award_output dictionary
    # del award_winners_output['Winners']

    # # Update the award_output dictionary with the single winner
    # award_winners_output['Winner'] = {
    #     "Name": winner,
    #     "Number of Tweets": winner_count
    # }

    # award_winners_final_output = {
    #     "Event": "Golden Globes 2013",
    #     "Host": "Tina Fey",
    #     "Awards": [
    #         award_winners_output
    #     ]
    # }

    # print("Human-readable output:")
    # print(format_human_readable(award_winners_final_output))
    
    # print("\nJSON output:")
    # print(json.dumps(award_winners_final_output, indent=4))

if __name__ == "__main__":
    # hosts, award_data = import_data()

    # print('PRESENTERS...')
    # presenters = get_presenters(award_data.keys())
    # print(presenters)

    main(2013)
    # print("AWARDS...")
    # print(find_award_names())
    
    # award_name = 'best director - motion picture'
    # print(f"AWARD: {award_name}")
    # get_award_winner(award_name=award_name, year=2013)







