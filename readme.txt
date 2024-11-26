GitHub Repo: https://github.com/zjerath/tweetminer

Steps to Run:
1. Upload the gg{year}.json file to the data folder. If you would like to test our code with hardcoded 
award names to see how it functions without cascading error, please upload gg{year}answers.json to the 
data folder as well.

2. Create a virtual environment with python=3.10. Activate the environment and run 'pip install -r requirements.txt'.
After that, run 'python -m spacy download en_core_web_lg'.

3. Call main with the necessary parameters, 'year' and 'use_hardcoded'. The 'use_hardcoded' parameter,
which defaults to False, is a boolean to decide whether or not to use hardcoded award names to avoid
cascading error. e.g. 'python main.py 2013 True' calls main with 2013 data and the answer award names.
'python main.py 2013' calls main with 2013 data and generates predictions based on the awards that we
found.

4. When all 'Processing Award' print statements have finished, both the human-readable and json outputs
will be printed to the console. The human-readable output also contains info on our additional task, where
we analyzed sentiments regarding the red carpet (best dressed, worst dressed, most controversially dressed).

File Structure:
The output directory contains json and human-readable outputs for both hardcoded and our found award names. 
Our main.py file contains the script used to run our program. It references multiple files in the util_functions
directory, which we use for preprocessing, prediction, aggregation, cross-checking against an external 
dataset (which we had to remove in order to save time, described below), and our additional task. The data 
directory is where all necessary json files should be placed (gg{year}.json or gg{year}answers.json to run 
main with use_hardcoded=True). Last of all, our notebooks directory contains jupyter notebooks with our 
initial logic and coding for several parts of the project. The Movies Dataset notebook was initially created
for our cross-checking, however we removed that and thus the 2 files that the notebook relies on as well. The
files to run that notebook were downloaded from kaggle.