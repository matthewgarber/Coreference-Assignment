#!usr/bin/bash
#
# Runs the entire coreference resolution pipeline on both the dev and test sets.
#
# Usage:
#	sh pipeline.sh
#

# Train classifier and perform coreference resolution.
python project3.py conll-2012/dev out -c save -v save

# Concatenate corpora.
sh scripts/concatenate_dev.sh
sh scripts/concatenate_output.sh

# Test and score on dev and output the results.
sh scripts/score.sh full_dev.gold_conll full_output.out_conll > dev_results.txt

rm -r out

# Load classifier and perform coreference resolution.
python project3.py conll-2012/test out -c load -v load

# Concatenate corpora.
sh scripts/concatenate_test.sh
sh scripts/concatenate_output.sh

# Test and score on test and output the results.
sh scripts/score.sh full_test.gold_conll full_output.out_conll > test_results.txt