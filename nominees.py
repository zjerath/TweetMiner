from util_functions.preprocessing_utils import preprocess_tweets
from util_functions.predictions_utils import extract_all_winners, extract_all_hosts, extract_all_award_names, extract_all_nominees
from util_functions.aggregation_utils import aggregate_entities, format_human_readable, named_entity_recognition

# example preprocessing in main
df = preprocess_tweets("data/gg2013.json")

# awards list
awards = ["best screenplay - motion picture", "best director - motion picture", 
          "best performance by an actress in a television series - comedy or musical",
          "best foreign language film",
          "best performance by an actor in a supporting role in a motion picture",
          "best performance by an actress in a supporting role in a series, mini-series or motion picture made for television",
          "best motion picture - comedy or musical",
          "best performance by an actress in a motion picture - comedy or musical",
          "best mini-series or motion picture made for television",
          "best original score - motion picture",
          "best performance by an actress in a television series - drama",
          "best performance by an actress in a motion picture - drama",
          "cecil b. demille award",
          "best performance by an actor in a motion picture - comedy or musical",
          "best motion picture - drama",
          "best performance by an actor in a supporting role in a series, mini-series or motion picture made for television",
          "best performance by an actress in a supporting role in a motion picture",
          "best television series - drama",
          "best performance by an actor in a mini-series or motion picture made for television",
          "best performance by an actress in a mini-series or motion picture made for television",
          "best animated feature film",
          "best original song - motion picture",
          "best performance by an actor in a motion picture - drama",
          "best television series - comedy or musical",
          "best performance by an actor in a television series - drama",
          "best performance by an actor in a television series - comedy or musical"
          ]

def find_nominees(df, awards):
    # takes in the preprocessed df and hard-coded list of awards
    top_nominees_by_award = []
    for award in awards:
        award_nominees = extract_all_nominees(df, award)
        # Get the top 5 nominees
        top_nominees = award_nominees["Nominees"][:5]
        # Store the result in the list
        top_nominees_by_award.append({
            "Award": award,
            "Top 5 Nominees": top_nominees
        })
    return top_nominees_by_award