import sklearn
from mentions import Mention, MentionPair
from sklearn.feature_extraction import DictVectorizer

class CoreferenceClassifier:
    """A wrapper for scikit-learn classifiers, with methods for performing
    coreference resolution on TestDocument objects.
    """

    def __init__(self, classifier):
        """Initializes a new CoreferenceClassifier object.

        Args:
            classifier: An instantiated scikit-learn classifier
        """
        self.classifier = classifier

    def train(self, mention_pair_vectors, label_vector):
        """Trains the classifier on the given data.

        Args:
            mention_pair_vectors: A numpy matrix or scipy sparse matrix of
                document feature vectors, with the shape:
                (# of mention pair samples, # of features).
            label_vector: A numpy vector or scipy sparse vector of mention pair
                labels, with the shape (# of mention pair samples).
        """
        self.classifier.fit(mention_pair_vectors, label_vector)

    def do_coreference_resolution(self, conll_file_docs, vectorizer):
        """
        Performs coreference resolution on each document in the given file,
        determining the most likely coreference cluster for each mention in
        each document.

        The algorithm for selecting coreference chains/clusters is the same
        pairwise algorithm used in "A Machine Learning Approach to Coreference
        Resolution of Noun Phrases" by Soon, Ng, and Lim. For each mention j
        after the first, it is paired with the closest antecedent i, followed
        by the next closest, and so on, until the classifier has labeled a pair
        as coreferring or until the are no more preceding mentions.

        Args:
            conll_file_docs: A list of TestDocument objects, all from the same
                source file.
            vectorizer: A DictVectorizer trained to convert dicts derived from
                MentionPair objects into vectors.
        Returns:
            The given list of documents with each document's Mentions labeled.
        """
        for doc in conll_file_docs:
            mentions = doc.mentions
            for mention in mentions:
                mention.label = None
            coref_id = 0
            for j in range(1, len(mentions)):
                mention_j = mentions[j]
                for i in reversed(range(j)):
                    mention_i = mentions[i]
                    pair = MentionPair(mention_i, mention_j)
                    pair_vector = vectorizer.transform(pair.to_dict())
                    label = self.classifier.predict(pair_vector)[0]
                    if label == 1:
                        if mention_i.label == None:
                            mention_i.label = coref_id
                            mention_j.label = coref_id
                            coref_id += 1
                        else:
                            mention_j.label = mention_i.label
                        break
            # For any mention not assigned a coreference group, assign an
            # individual coreference ID.
            for mention in mentions:
                if mention.label == None:
                    mention.label = coref_id
                    coref_id += 1
        return conll_file_docs
        
