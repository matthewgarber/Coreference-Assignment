import codecs
import os
import re
from os.path import join as pjoin
from mentions import Mention, MentionPair

class Document:
    """A container for storing all of the mention pairs extracted from a file
    in CONLL 2012 format. This class is used for training and should not be used
    for testing.

    Attributes:
        mention_pairs: A list of MentionPair objects extracted from the given
            text.
        labels: A list of booleans corresponding to each mention pair, denoting
            whether or not each pair of mentions corefer.
        file_id: The filepath of the corresponding CONLL file, relative to the
            'annotations' subdirectory in the corpus.
        part: The number (as a string) of this individual document within the
            CONLL file.
    """

    def __init__(self, doc_text):
        """Initializes a new Document object from some text in CONLL 2012
        format.

        Args:
            doc_text: A single CONLL 2012 document, as a string.
        """

        sents = self._parse_doc_text(doc_text)
        mentions = self._get_mentions(sents)
        pairs, labels = self._generate_mention_pairs_and_labels(mentions)
        self.mention_pairs = pairs
        self.labels = labels

    def _parse_doc_text(self, doc_text):
        """Parses the CONLL document and returns the document's sentences,
        including information extracted from each row of the CONLL document.

        Args:
            doc_text: A single CONLL 2012 document, as a string.
        Returns:
            A list of sublists of dictionaries. Each dictionary corresponds to
            a single row of the document and stores some of the information
            stored in that row.
        """
        header_pattern = '#begin document \(([\w\d/]*)\); part (\d+)'
        sents = []
        for line in doc_text.split('\n'):
            header_match = re.match(header_pattern, line)
            if header_match:
                self.file_id = header_match.group(1)
                self.part = header_match.group(2)
            elif line != '\n' and line != '' and not line.startswith('#'):
                vals = self._row_to_dict(line.split())
                if vals['word_num'] == '0':
                    sents.append([vals])
                else:
                    sents[-1].append(vals)
        return sents
    
    def _row_to_dict(self, values):
        """Converts the given row values into a dictionary.

        The following information from each row is kept: The number of the word
        in the sentence, the word itself, its POS, the corresponding portion
        of the parse tree, the word's lemma, its sense, its named entity type,
        and its coreference information.

        Args:
            values: A list of values parsed from a single row of a CONLL 2012
            document.
        Returns:
            A dictionary storing some of the information contained in the given
            row.
        """
        _, _, word_num, word, pos, parse = values[:6]
        lemma, _, sense, _, ne = values[6:11]
        coref = values[-1]
        value_dict = {'word_num': word_num,
                      'word': word,
                      'pos': pos,
                      'parse': parse,
                      'lemma': lemma,
                      'sense': sense,
                      'ne': ne,
                      'coref': coref
                      }
        return value_dict

    def _get_mentions(self, sents):
        """Extracts the mentions from the given sentences.

        Args:
            sents: A list of sublists of dictionaries corresponding to rows of
            the CONLL document.
        Returns:
            A list of Mention objects in the order that they appear in the
            document.
        """
        mentions = []
        word_offset = 0
        # Iterate over sentences in the doc.
        for i in range(len(sents)):
            if i != 0:
                word_offset += len(sents[i-1])
            # Iterate over words/rows in the sentence.
            sent = sents[i]
            for j in range(len(sent)):
                vals = sent[j]
                coref_parts = vals['coref'].split('|')
                for part in coref_parts:
                    singleton_match = re.match('^\((\d+)\)$', part)
                    multi_match = re.match('^\((\d+)$', part)
                    if singleton_match:
                        label = singleton_match.group(1)
                        new_mention = Mention(j, j+1, sents[i], i, word_offset, label=label)
                        mentions.append(new_mention)
                    elif multi_match:
                        label = multi_match.group(1)
                        mention_end = part[1:] + ')'
                        k = j + 1
                        found_end = False
                        while not found_end:
                            coref = sent[k]['coref']
                            if mention_end in coref.split('|'):
                                found_end = True
                            else:
                                k += 1
                        new_mention = Mention(j, k+1, sents[i], i, word_offset, label=label)
                        mentions.append(new_mention)
        return mentions

    def _generate_mention_pairs_and_labels(self, mentions):
        good_pairs = []
        bad_pairs = []
        for i in range(len(mentions)):
            mention_i = mentions[i]
            for j in range(i+1, len(mentions)):
                mention_j = mentions[j]
                new_pair = MentionPair(mention_i, mention_j)
                if mention_i.label == mention_j.label:
                    good_pairs.append(new_pair)
                    break
                else:
                    bad_pairs.append(new_pair)
        pairs = good_pairs + bad_pairs
        labels = [1] * len(good_pairs) + [0] * len(bad_pairs)
        return pairs, labels
                    

class TestDocument(Document):
    """A container for storing mentions extracted from a file in CONLL 2012
    format. This Document subclass is used for testing, where it is more
    efficient to compute MentionPairs on the fly.

    Attributes:
        sents: A list of sublists of dictionaries corresponding to rows of
            the CONLL document.
        mentions: A list of Mention objects extracted from the given text.
        file_id: The filepath of the corresponding CONLL file, relative to the
            'annotations' subdirectory in the corpus.
        part: The number (as a string) of this individual document within the
            CONLL file.
    """

    def __init__(self, doc_text):
        """Initializes a new TestDocument object.

        Args:
            doc_text: A single CONLL 2012 document, as a string.
        """
        self.sents = self._parse_doc_text(doc_text)
        self.mentions = self._get_mentions(self.sents)

    def to_conll_format(self):
        """Outputs the data stored in to document into CONLL 2012 format.

        Returns:
            The document text in CONLL 2012 format, as a string.
        """
        self._change_coref_values()
        lines = []
        header = '#begin document ({}); part {}\n'.format(self.file_id, self.part)
        lines.append(header)
        for i in range(len(self.sents)):
            sent = self.sents[i]
            for row_dict in sent:
                row_vals = [self.file_id,
                            str(int(self.part)),
                            row_dict['word_num'],
                            row_dict['word'],
                            row_dict['pos'],
                            row_dict['parse'],
                            row_dict['lemma'],
                            '-',
                            row_dict['sense'],
                            '-',
                            row_dict['ne'],
                            '-',
                            row_dict['coref']
                            ]
                line = '\t'.join(row_vals) + '\n'
                lines.append(line)
            lines.append('\n')
        lines.append('#end document\n')
        return ''.join(lines)

    def _change_coref_values(self):
        """Changes the coreference values of each row in the document so that
        they correctly correspond with that coreference cluster IDs of the
        documents mentions.
        """
        # Zero out labels
        for sent in self.sents:
            for row_dict in sent:
                row_dict['coref'] = '-'
        for mention in self.mentions:
            if mention.start == mention.end - 1:
                self._set_coref(mention.sent_id,
                                mention.start,
                                mention.label,
                                'singleton'
                                )
            else:
                self._set_coref(mention.sent_id,
                                mention.start,
                                mention.label,
                                'start'
                                )
                self._set_coref(mention.sent_id,
                                mention.end - 1,
                                mention.label,
                                'end'
                                )

    def _set_coref(self, sent_num, word_num, coref_id, pos):
        """Sets the 'coref' value of the row corresponding to the given sentence
        number and word number to the appropriate coreference cluster ID in the
        appropriate format.

        Args:
            sent_num: An integer denoting the number of the sentence where the
                'coref' value to change is located.
            word_num: An integer denoting the number of the word where the
                'coref' value to change is located.
            coref_id: A string denoting the coreference cluster ID to set the
                'coref' value to.
            pos: The position to put the coref ID in, if there are embedded
                mentions. Should be either 'start', 'end', or 'singleton'.
        """
        row_dict = self.sents[sent_num][word_num]
        old_coref = row_dict['coref']
        new_coref = None
        if pos == 'singleton':
            new_coref = '({})'.format(coref_id)
        elif pos == 'start':
            new_coref = '({}'.format(coref_id)
        elif pos == 'end':
            new_coref = '{})'.format(coref_id)
        if old_coref == '-':
            row_dict['coref'] = new_coref
        elif pos == 'start' or (pos == 'singleton' and old_coref.startswith('(')):
            row_dict['coref'] = '{}|{}'.format(old_coref, new_coref)
        elif pos == 'end' or (pos == 'singleton' and not old_coref.startswith('(')):
            row_dict['coref'] = '{}|{}'.format(new_coref, old_coref)


class ConllCorpus:
    """A corpus of documents in CONLL 2012 format, storing coreference
    information from those documents. The documents are only accessible through
    a generator, which is necessary with large corpora to conserve memory.

    Atrributes:
        corpus_files: A list of string filepaths to all of the files in the
            corpus.
        doc_class: The class used as a representation of each document.
        doc_generator: A generator that generates a list of sublists of
            documents. Each sublist corresponds to a CONLL-formatted file and
            contains each document stored in that file.
    """

    def __init__(self, corpus_path, doc_class=Document):
        """Initializes a new ConllCorpus object.

        Args:
            corpus_path: A string denoting the directory of the corpus.
            doc_class: The class used as a representation of each document.
        """
        self.corpus_files = self._get_corpus_files(corpus_path)
        self.doc_class = doc_class
        self.doc_generator = (self._file_to_docs(f, self.doc_class)
                              for f in self.corpus_files
                              )

    def reset_generator(self):
        """Resets the doc_generator so that the corpus can be iterated over
        again, without having to instantiate a new ConllCorpus object.
        """
        self.doc_generator = (self._file_to_docs(f, self.doc_class)
                              for f in self.corpus_files
                              )

    def _get_corpus_files(self, corpus_path):
        """Retrieves the filenames of the all of the relevant document files
        in the given corpus directory.

        Args:
            corpus_path: A string denoting the directory of the corpus.
        Returns:
            corpus_files: A list of string filepaths to all of the files in the
                corpus.
        """
        corpus_files = []
        for root, dirs, files in os.walk(corpus_path):
            for filename in files:
                if corpus_path.endswith('test'):
                    if filename.endswith('gold_conll'):
                        corpus_files.append(pjoin(root, filename))
                else:
                    if filename.endswith('auto_conll'):
                        corpus_files.append(pjoin(root, filename))
        return corpus_files

    def _file_to_docs(self, filename, doc_class):
        """Converts the file with the given filename into a list of objects of
        the given class.

        Args:
            filename: The name of the document file.
            doc_class: The class used as a representation of each document.
        Returns:
            A list of objects of the given doc_class.
        """
        conll_file = codecs.open(filename, encoding='utf8')
        file_text = conll_file.read()
        file_docs = re.findall('#begin.*?#end document', file_text, flags=re.DOTALL)
        processed_docs = []
        for doc in file_docs:
            processed_docs.append(doc_class(doc))
        return processed_docs

