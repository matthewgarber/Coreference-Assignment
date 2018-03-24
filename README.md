# Coreference-Assignment
Name: Matthew Garber
Term: Spring 2017
Class: Information Extraction
Assignment #3: Coreference Resolution

To run the full pipeline program:
	
	sh pipeline.sh <save-or-load>
	
If <save-or-load> is 'save', then a new DictVectorizer will be fitted and saved
as 'vectorizer.pickle' and a new classifier will be trained and saved as
'classifier.pickle'.

If <save-or-load> is 'load' then the DictVectorizer will be loaded from 
'vectorizer.pickle' and a the classifier will be loaded from 'classifier.pickle'.

In either case, the classifier will be scored on the dev corpus and the test
corpus. Results of the dev set will be written to 'dev_results.txt' and results
of the test set will be written to 'test_results.txt'.

## Directory Structure ##
The program expects the following directory structure:

	-conll-2012 : The uncompressed CONLL 2012 corpus.

	-names : A collection of files listing the counts and genders of the names
			 of people born in the U.S from 1880 to 2015.
			 
	-out : The output of the coreference resolution program.
	
	-reference-coreference-scorers-8.01 : Contains a scoring program.
	
	-scripts
		-concatenate_dev.sh : Combines the dev corpus into a single file called
							  full_dev.gold_conll
		-concatenate_output.sh : Combines the outputted corpus (located in out/) into
								 a single file called full_output.out_conll
		-concatenate_test.sh : Combines the test corpus into a single file called
							   full_test.sh
		-score.sh : Scores a given output corpus against a given gold corpus.
		
	-classifier.py : Contains a CoreferenceClassifier class. See the file's
					 documentation for more details.
					 
	-corpus.py : Contains classes for handling CONLL files, documents, and corpora.
				 See the file's documentation for more details.
				 
	-mentions.py : Contains classes for handling mentions and pairs of mentions
				   in a CONLL document. See the file's documentation for more
				   details.
				   
	-pipeline.sh : A script that runs the entire coreference resolution pipeline
				   and outputs the scores on the dev and test corpora.

	-project3.py : Trains or load a classifier and classify a given corpus. See
				   documentation for usage and details.