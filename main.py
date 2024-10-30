import json
import pandas as pd
from util_functions.preprocessing_utils import preprocess_tweets
from util_functions.predictions_utils import extract_winners, extract_all_winners, extract_all_hosts, extract_all_award_names, extract_all_nominees, extract_all_presenters, extract_potential_nominees
from util_functions.aggregation_utils import aggregate_candidates, aggregate_entities, format_human_readable, named_entity_recognition, define_entities, is_person_name
from util_functions.sentiment_analysis_utils import analyze_best_worst_dressed

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

    # Process each host entry to merge partial names with full names
    host_counts = {}
    for name, count in significant_hosts:
        added = False
        for existing_name in list(host_counts.keys()):
            if name in existing_name or existing_name in name:
                host_counts[existing_name] += count
                added = True
                break
        if not added:
            host_counts[name] = host_counts.get(name, 0) + count

    # Sort the hosts by count in descending order
    sorted_hosts = sorted(host_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_hosts[:2] if len(sorted_hosts) > 2 else sorted_hosts
    
    
    # # Get the hosts that are mentioned with frequency above the threshold
    # significant_hosts = [entity for entity, count in hosts_entities.items() if count > threshold]

    # # If no hosts meet the threshold, return the most mentioned entity
    # if not significant_hosts:
    #     sorted_candidates = sorted(hosts_entities.items(), key=lambda x: x[1], reverse=True)
    #     significant_hosts = [entity for entity, _ in sorted_candidates[:1]]
    
    # return significant_hosts

def find_nominees(df, award, top_n):
    # takes in the preprocessed df and hard-coded list of awards
    #top_nominees_by_award = []
    #for award in awards:
    award_nominees = extract_all_nominees(df, award)
    # Get the top 6 nominees
    top_nominees = award_nominees["Nominees"][:top_n]
    # Store the result in the list
    '''top_nominees_by_award.append({
        "Award": award,
        f"Top {top_n} Nominees": top_nominees
    })'''
    return top_nominees
    
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

def get_award_winner(df, award, nominees):
    potential_award_winners = extract_winners(df, award, nominees)

    candidate_dict = {winner["Name"]: winner["Number of Tweets"] for winner in potential_award_winners["Winners"]}
    candidate_dict = dict(sorted(candidate_dict.items(), key=lambda item: item[1], reverse=True))

    potential_winners = list(candidate_dict.keys())[:6]

    # Find winner that exists in both nominee list and potential winners
    valid_winners = [w for w in potential_winners if w in nominees]
    
    if valid_winners:
        # Get winner with highest count from valid winners
        winner = max(valid_winners, key=lambda x: candidate_dict[x])
        # Ensure winner is in nominees list
        if winner not in nominees:
            nominees.append(winner)
    else:
        # Fallback if no valid winner found
        winner = potential_winners[0] if potential_winners else "Unknown"

    return winner

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

'''def get_presenters(df, award_names, hosts):
    presenters = []
    for award in award_names:
        award_presenters = get_award_presenters(df, award, hosts)
        presenters.append({
            "Award Name": award,
            "Presenters": [presenter['Name'] for presenter in award_presenters]
        })

    return presenters'''

# Function to use a hardcoded list of the awards and nominees to avoid cascading error
'''This function DOES NOT output award names found by us. To see the answers with our
generated award names included, must use the cascading_output function''' 
def hardcoded_output(df, hardcoded_award_names):
    # Hosts
    print("Processing Hosts")
    hosts = find_hosts(df)
    host_names = [host[0] for host in hosts]

    # Format outputs
    human_readable_output = "Hosts: " + ", ".join(host_names) + "\n\n"
    json_output = {
        "hosts": host_names,
        "award_data": {}
    }

    # Loop through awards
    for award_name in hardcoded_award_names:
        print(f"Processing Award: {award_name}")
        
        # Presenters
        award_presenters = get_award_presenters(df, award_name, host_names)
        presenter_names = [presenter['Name'] for presenter in award_presenters]

        # Nominees
        nominee_names = find_nominees(df, award_name, 6)

        # Winner
        winner = get_award_winner(df, award_name, nominee_names)
        
        # Add to human-readable output
        human_readable_output += f"Award: {award_name}\n"
        human_readable_output += "Presenters: " + ", ".join(presenter_names) + "\n"
        human_readable_output += "Nominees: " + ", ".join(nominee_names) + "\n\n"
        human_readable_output += f"Winner: {winner}\n\n"
        
        # Add to JSON output
        json_output["award_data"][award_name] = {
            "Presenters": presenter_names,
            # Placeholder for nominees and winner, to be filled later
            "Nominees": nominee_names,
            "Winner": winner
        }
    
    # Red carpet
    red_carpet_results = analyze_best_worst_dressed(df)

    # Extract only the names
    best_dressed_names = [person["Name"] for person in red_carpet_results["Best Dressed"]]
    worst_dressed_names = [person["Name"] for person in red_carpet_results["Worst Dressed"]]
    controversial_dressed_names = [person["Name"] for person in red_carpet_results["Most Controversial"]]

    # Add to human-readable output
    human_readable_output += "Best Dressed: " + ", ".join(best_dressed_names) + "\n"
    human_readable_output += "Worst Dressed: " + ", ".join(worst_dressed_names) + "\n"
    human_readable_output += "Most Controversially Dressed: " + ", ".join(controversial_dressed_names) + "\n"

    print(f"Human-readable format:\n{human_readable_output}")
    print(f"JSON format:\n{json.dumps(json_output, indent=4)}")

# Function to use our generated list of the awards and nominees to view effects of cascading error
def cascading_output(df):
    # Hosts
    print("Finding Hosts...")
    hosts = find_hosts(df)
    host_names = [host[0] for host in hosts]

    # Format outputs
    human_readable_output = "Hosts: " + ", ".join(host_names) + "\n"
    json_output = {
        "hosts": host_names,
        "award_data": {}
    }

    print(f"Human-readable format:\n{human_readable_output}")
    print(f"JSON format:\n{json.dumps(json_output, indent=4)}")

def main(year):
    # this is for hard-coded stuff to prevent cascading error
    with open(f"data/gg{year}answers.json", 'r') as f:
        answers_data = json.load(f)
    hardcoded_awards_data = answers_data['award_data']
    hardcoded_award_names = list(hardcoded_awards_data.keys())
    # hardcoded_award_names = [name for name in hardcoded_award_names if 'best' in name]
    # hardcoded_nominees = [hardcoded_awards_data[award]['nominees'] + [hardcoded_awards_data[award]['winner']] for award in hardcoded_award_names]
    
    df = preprocess_tweets(f"data/gg{year}.json")

    print(f"nominees for {hardcoded_award_names[0]}")
    nominees = find_nominees(df, hardcoded_award_names[0], 6)
    print(nominees)
    
    hardcoded_output(df, hardcoded_award_names)

    # print("AWARD NAMES...")
    # awards = find_award_names(df)
    # print(awards)

    # '''
    # # using our award & host predictions to retrieve presenters
    # print("PRESENTERS...")
    # presenters = get_presenters(df, awards, hosts)
    # print(presenters)
    # '''

    # print("PRESENTERS...")
    # presenters = get_presenters(df, hardcoded_award_names, hardcoded_hosts_data)
    # print(presenters)

    '''
    # using our award & host predictions to retrieve nominees
    print("NOMINEES...")
    nominees = find_nominees(df, awards, 5)
    print(nominees)
    '''

    print("NOMINEES...")
    # x = df['clean_text'].apply(lambda x: extract_potential_nominees(x, hardcoded_award_names[0]))
    # print(x)
    nominees = find_nominees(df, hardcoded_award_names, 5)
    print(nominees)

    # '''
    # # using our award & nominee predictions to retrieve winners
    # print("WINNERS...")
    # winners = find_nominees(df, awards, 5)
    # print(winners)
    # '''

    # print("NOMINEES...")
    # nominees = find_nominees(df, hardcoded_award_names, 5)
    # print(nominees)

    print(analyze_best_worst_dressed(df))
    
    award_winners_output = get_award_winners(df, hardcoded_award_names, hardcoded_nominees)
    print(award_winners_output)

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







