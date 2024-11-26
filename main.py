import os
import json
import sys
import pandas as pd
from util_functions.preprocessing_utils import preprocess_tweets
from util_functions.predictions_utils import extract_winners, extract_all_hosts, extract_all_award_names, extract_all_nominees, extract_all_presenters
from util_functions.aggregation_utils import aggregate_entities, named_entity_recognition, is_person_name
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
    nominees = award_nominees["Nominees"]
    nominee_names = [nominee["Name"] for nominee in nominees]
    top_nominees = nominee_names[:top_n]
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


# Function to process awards given award names and host names
def process_awards(df, award_names, host_names):
    human_readable_output = ""
    json_output = {"award_data": {}}
    # Loop through awards
    for award_name in award_names:
        print(f"Processing Award: {award_name}")
        # Presenters
        award_presenters = get_award_presenters(df, award_name, host_names)
        presenter_names = [presenter['Name'] for presenter in award_presenters]
        # Nominees
        nominee_names = find_nominees(df, award_name, 6)
        # Winner
        winner = get_award_winner(df, award_name, nominee_names)
        # Format for output
        human_readable_output += (
            f"Award: {award_name}\nPresenters: {', '.join(presenter_names)}\n"
            f"Nominees: {', '.join(nominee_names)}\nWinner: {winner}\n\n"
        )
        json_output["award_data"][award_name] = {
            "Presenters": presenter_names,
            "Nominees": nominee_names,
            "Winner": winner
        }
    return human_readable_output, json_output

# Function to deal with extra task
def process_red_carpet(df):
    red_carpet_results = analyze_best_worst_dressed(df)
    # Extract only the names
    best_dressed_names = [person["Name"] for person in red_carpet_results["Best Dressed"]]
    worst_dressed_names = [person["Name"] for person in red_carpet_results["Worst Dressed"]]
    controversial_dressed_names = [person["Name"] for person in red_carpet_results["Most Controversial"]]
    # Format human-readable output
    human_readable_output = (
        f"Best Dressed: {', '.join(best_dressed_names)}\n"
        f"Worst Dressed: {', '.join(worst_dressed_names)}\n"
        f"Most Controversially Dressed: {', '.join(controversial_dressed_names)}\n"
    )
    return human_readable_output

# Funtion to save JSON and human-readable outputs to respective files."""
def save_output_files(json_output, human_output, file_prefix):
    json_file = f"output/{file_prefix}_answers.json"
    human_file = f"output/{file_prefix}_output.txt"
    # Write JSON output
    with open(json_file, 'w') as f:
        json.dump(json_output, f, indent=4)
    # Write human-readable output
    with open(human_file, 'w') as f:
        f.write(human_output)
    print(f"JSON output saved to {json_file}")
    print(f"Human-readable output saved to {human_file}")

# Function to use a hardcoded list of the awards and nominees to avoid cascading error
'''This function DOES NOT output award names found by us. To see the answers with our
generated award names included, must use the cascading_output function''' 
def hardcoded_output(df, hardcoded_award_names):
    print("Using hardcoded list of awards to avoid cascading error")
    # Hosts
    print("Processing Hosts")
    hosts = find_hosts(df)
    host_names = [host[0] for host in hosts]
    # Format outputs
    human_readable_output = "Hosts: " + ", ".join(host_names) + "\n\n"
    json_output = {"hosts": host_names}
    # Awards
    award_text, award_json = process_awards(df, hardcoded_award_names, host_names)
    # Add to outputs
    human_readable_output += award_text
    json_output.update(award_json)
    # Red Carpet
    human_readable_output += process_red_carpet(df)
    # Output
    save_output_files(json_output, human_readable_output, "hardcoded")
    print(f"Human-readable format:\n{human_readable_output}")
    print(f"JSON format:\n{json.dumps(json_output, indent=4)}")

# Function to use our generated list of the awards and nominees to view effects of cascading error
def cascading_output(df):
    print("Not using any hardcoded lists, might result in cascading error")
    # Hosts
    print("Processing Hosts")
    hosts = find_hosts(df)
    host_names = [host[0] for host in hosts]
    # Format outputs
    human_readable_output = "Hosts: " + ", ".join(host_names) + "\n\n"
    json_output = {"hosts": host_names}
    # Awards
    print("Extracting Awards")
    awards = find_award_names(df)
    award_names = list(set([award['Name'] for award in awards]))
    award_text, award_json = process_awards(df, award_names, host_names)
    # Add to outputs
    human_readable_output += award_text
    json_output.update(award_json)
    # Red carpet
    human_readable_output += process_red_carpet(df)
    # Output
    save_output_files(json_output, human_readable_output, "cascading")
    print(f"Human-readable format:\n{human_readable_output}")
    print(f"JSON format:\n{json.dumps(json_output, indent=4)}")

# To call main, use command 'python main.py {year} {bool}'
# e.g. 'python main.py 2013 True' calls main with 2013 data and hardcoded award names
def main(year, use_hardcoded=False):
    df = preprocess_tweets(f"data/gg{year}.json")
    os.makedirs("output", exist_ok=True)
    # If use_hardcoded, use the hardcoded award names to prevent cascading error
    if use_hardcoded:
        # This is for hard-coded stuff to prevent cascading error
        with open(f"data/gg{year}answers.json", 'r') as f:
            answers_data = json.load(f)
        hardcoded_awards_data = answers_data['award_data']
        hardcoded_award_names = list(hardcoded_awards_data.keys())
        hardcoded_output(df, hardcoded_award_names)
    # If nothing specified, use our raw implementation for everything
    else:
        cascading_output(df)

if __name__ == "__main__":
    # Set default values for year and use_hardcoded
    year = 2013
    use_hardcoded = False

    # Update year and use_hardcoded based on command-line arguments
    if len(sys.argv) > 1:
        year = int(sys.argv[1])  # First argument is the year
    if len(sys.argv) > 2:
        use_hardcoded = sys.argv[2].lower() == 'true'  # Second argument is True/False for hardcoded

    main(year, use_hardcoded=use_hardcoded)