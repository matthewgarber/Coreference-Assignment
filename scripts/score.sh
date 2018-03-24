#!usr/bin/bash
#
# Scores the given output corpus against the given gold corpus, showing every
# score type.
#
# Usage:
#	sh score.sh <gold-corpus-file> <output-corpus-file>
#

gold=$1
output=$2
perl reference-coreference-scorers-8.01/scorer.pl all $gold $output none