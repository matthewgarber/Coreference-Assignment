"""
Name: Matthew Garber
Term: Spring 2017
Class: COSI 134 Information Extraction
Assignment #3: Coreference Resolution

This program performs coreference resolution on CONLL formatted documents.
It accomplishes this using a pairwise Logistic Regression classifier. 

To run:
    python project3.py <test-corpus-dir> <output-dir>

    Options:
        -c or --classifier: 'save' to train and save the classifier, 'load' to
            load a previously trained classifier.
        -v or --vectorizer: 'save' to fit and save a DictVectorizer, 'load' to
            load a previously fitted DictVectorizer.

This program can take some time to run, since it must iterate multiple times
through the training corpus due to memory limitations.
"""

import codecs
import os
import pickle
import re
from argparse import ArgumentParser
from classifier import CoreferenceClassifier
from corpus import *
from mentions import Mention, MentionPair
from numpy import int8, matrix, zeros
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

def label_corpus(classifier, vectorizer, corpus, output_dir):
    """Performs coreference resolution on the given corpus with the given
    classifier and writes the corresponding files to the output directory.

    Args:
        classifier: A trained CoreferenceClassifier object.
        vectorizer: A fitted DictVectorizer.
        corpus: A ConllCorpus of TestDocument objects.
        output_dir: The directory to write the labeled files to.
    """
    for conll_file in corpus.doc_generator:
        labelled_file = classifier.do_coreference_resolution(conll_file, vectorizer)
        subpath = '{}.{}'.format(labelled_file[0].file_id, 'out_conll')
        filepath = pjoin(output_dir, subpath)
        file_text = ''.join([doc.to_conll_format() for doc in labelled_file])
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        with codecs.open(filepath, 'wb', encoding='utf8') as output_file:
            output_file.write(file_text)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('test_corpus_dir')
    parser.add_argument('output_dir')
    parser.add_argument('-c', '--classifier', choices=['save', 'load'])
    parser.add_argument('-v', '--vectorizer', choices=['save', 'load'])
    args = parser.parse_args()

    train_corpus = ConllCorpus('conll-2012/train',
                               doc_class=Document
                               )

    complete_dict_generator = (mention_pair.to_dict()
                               for f in train_corpus.doc_generator
                               for doc in f
                               for mention_pair in doc.mention_pairs
                               )
    vectorizer = None
    if args.vectorizer == 'save':
        print('Fitting DictVectorizer...')
        vectorizer = DictVectorizer(dtype=int8)
        vectorizer.fit(complete_dict_generator)
        with open('vectorizer.pickle', 'wb') as f:
            pickle.dump(vectorizer, f)
    elif args.vectorizer == 'load':
        with open('vectorizer.pickle', 'rb') as f:
            vectorizer = pickle.load(f)

    corpus_paths = [r'conll-2012/train/english/annotations/bc',
                    r'conll-2012/train/english/annotations/bn',
                    r'conll-2012/train/english/annotations/mz',
                    r'conll-2012/train/english/annotations/nw',
                    r'conll-2012/train/english/annotations/pt',
                    r'conll-2012/train/english/annotations/tc',
                    r'conll-2012/train/english/annotations/wb'
                    ]

    classifier = None
    if args.classifier == 'save':
        X = None
        y = []
        print('Creating mention pair vectors...')
        for corpus_path in corpus_paths:
            print('Working on {}...'.format(corpus_path))
            train_corpus = ConllCorpus(corpus_path)
            for f in train_corpus.doc_generator:
                pair_dicts = [mention_pair.to_dict()
                              for doc in f
                              for mention_pair in doc.mention_pairs
                              ]
                labels = [label for doc in f for label in doc.labels]
                if pair_dicts != []:
                    if X == None:
                        X = vectorizer.transform(pair_dicts)
                    else:
                        X2 = vectorizer.transform(pair_dicts)
                        X = vstack((X, X2), dtype=int8)
                    if y == None:
                        y = csr_matrix(labels)
                    else:
                        y2 = csr_matrix(labels)
                        y = hstack((y, y2), dtype=int8)
        classifier = CoreferenceClassifier(LogisticRegression())
        new_y = y.toarray()[0]
        print('Training classifier...')
        classifier.train(X, new_y)
        with open('classifier.pickle', 'wb') as f:
            pickle.dump(classifier, f)
    elif args.classifier == 'load':
        print('Loading classifier...')
        with open('classifier.pickle', 'rb') as f:
            classifier = pickle.load(f)

    test_corpus = ConllCorpus(args.test_corpus_dir, doc_class=TestDocument)
    print('Performing coreference resolution and writing results...')
    label_corpus(classifier, vectorizer, test_corpus, args.output_dir)

