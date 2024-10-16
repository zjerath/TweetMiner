import json
from preprocessing_utils import preprocess_tweets
from predictions_utils import extract_all_winners, extract_all_hosts
from aggregation_utils import aggregate_entities, format_human_readable, named_entity_recognition

def find_hosts():
    df = preprocess_tweets("gg2013.json")

    hosts_tweets = extract_all_hosts(df)
    hosts_entities = named_entity_recognition(hosts_tweets)

    hosts_entities = aggregate_entities(hosts_entities)

    print(hosts_entities)

    # Get the hosts that are mentioned with very high frequency (30x median frequency)
    sorted_counts = sorted(hosts_entities.values())
    median_frequency = sorted_counts[len(sorted_counts) // 2] if len(sorted_counts) % 2 != 0 else (sorted_counts[len(sorted_counts) // 2 - 1] + sorted_counts[len(sorted_counts) // 2]) / 2
    median_frequency += 1
    significant_hosts = [entity for entity, count in hosts_entities.items() if count > 30 * median_frequency]
    return significant_hosts if significant_hosts else [max(hosts_entities, key=hosts_entities.get)]


def main():

    df = preprocess_tweets("gg2013.json")

    award_winners_output = extract_all_winners(df, "Best Picture", nominees=[], presenters=[])

    award_winners_output['Winners'] = aggregate_entities(award_winners_output['Winners'])

    # Get the winner with the highest value
    winner = max(award_winners_output['Winners'], key=award_winners_output['Winners'].get)

    award_winners_output['Winner'] = winner
    
    # Get the count of mentions for the winner
    winner_count = award_winners_output['Winners'][winner]

    # Remove the 'Winners' key from the award_output dictionary
    del award_winners_output['Winners']

    # Update the award_output dictionary with the single winner
    award_winners_output['Winner'] = {
        "Name": winner,
        "Number of Tweets": winner_count
    }

    award_winners_final_output = {
        "Event": "Golden Globes 2013",
        "Host": "Tina Fey",
        "Awards": [
            award_winners_output
        ]
    }

    print("Human-readable output:")
    print(format_human_readable(award_winners_final_output))
    
    print("\nJSON output:")
    print(json.dumps(award_winners_final_output, indent=4))

if __name__ == "__main__":
    print(find_hosts())







