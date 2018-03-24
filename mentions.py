import os
import re
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.corpus.reader.wordnet import WordNetError

# Synsets frequently used in Mention and MentionPair feature extraction.
MALE_SYNSET = wordnet.synset('male.n.02')
MALE_WORDS = {'male', 'man', 'boy'}
FEMALE_SYNSET = wordnet.synset('female.n.02')
FEMALE_WORDS = {'female', 'woman', 'girl'}
PERSON_SYNSET = wordnet.synset('person.n.01')

VAGUE_SYNSETS = {wordnet.synset('entity.n.01'),
                 wordnet.synset('abstraction.n.06'),
                 wordnet.synset('physical_entity.n.01'),
                 wordnet.synset('object.n.01'),
                 wordnet.synset('artifact.n.01'),
                 wordnet.synset('whole.n.02'),
                 wordnet.synset('group.n.01')
                 }

# Dictionary used to determine the gender of pronoun mentions.
GENDER_DICT = {'she': 'female',
               'her': 'female',
               'herself': 'female',
               'he': 'male',
               'him': 'male',
               'himself': 'male',
               'his': 'male'
               }

# Dictionary used to determine the number of pronoun mentions.
PLURAL_PRONOUNS = {'they' 'their', 'them', 'themselves', 'these', 'those',
                       'we', 'us','our', 'ourselves'
                   }

# Sets used to determine the gender of named entities.
MALE_HONORIFICS = {'mr', 'mr.', 'mister', 'sir', 'lord'}
FEMALE_HONORIFICS = {'mrs', 'mrs.', 'ms', 'ms.', 'miss', 'lady'}

# Set of demonstrative pronouns/articles.
DEMONSTRATIVES = {'this', 'that', 'these', 'those'}

# Set of English stopwords.
STOPWORDS = stopwords.words('english')

def increment_count(count_dict, key, count):
    """Increments the count of the key in the given dictionary by the given
    amount.
    """
    count_dict[key] = count_dict.get(key, 0) + count

def make_first_name_dict(name_dir, start_year=1940, end_year=2016):
    """Creates a dictionary mapping first names to the gender that most
    frequently has that name.

    Args:
        name_dir: Directory of text files listing names, gender, and counts for
            each year.
        start_year: Year to start counting names from.
        end_year: Year to end counting names on.
    Returns:
        A dictionary mapping first names to their most frequent gender.
    """
    if start_year < 1880 or end_year > 2016:
        raise Exception("Years must be between 1880 and 2016")
    else:
        male_counts = {}
        female_counts = {}
        for year in range(start_year, end_year):
            filename = ''.join(['yob', str(year), '.txt'])
            with open(os.path.join(name_dir,filename)) as name_file:
                name_lines = name_file.readlines()
                for line in name_lines:
                    if line != '\n':
                        name, gender, num = line.split(',')
                        if gender == 'F':
                            increment_count(female_counts, name.lower(), int(num))
                        else:
                            increment_count(male_counts, name.lower(), int(num))
        name_dict = {}
        for name in male_counts:
            if name in female_counts:
                if male_counts[name] > female_counts[name]:
                    name_dict[name] = 'male'
                else:
                    name_dict[name] = 'female'
                del female_counts[name]
            else:
                name_dict[name] = 'male'
        for name in female_counts:
            # female_counts now only has female-only names.
            name_dict[name] = 'female'
        return name_dict

NAME_DICT = make_first_name_dict('names')



class Mention:
    """Objects of this class represent single mentions in as derived from a
    file in CONLL 2012 format. It possess methods to extract features from the
    mention, as well as its context, that can be later be used to derive
    pairwise features for coreference resolution.

    Attributes:
        label: The coreference cluster of this mention, as a string.
        start: The start index of the mention's span, as an int.
        end: The end index of the mention's span, as an int.

        string: The full text of the mention.
        head: The head word of the mention, as a string.
        head_pos: The part-of-speech of the mention, as a string.
        head_lemma: The lemma of the head word, as a string.
        head_sense: The number of the word sense of the head lemma in WordNet,
            as a string.
        head_i: The int index of the head word in the sentence.
        head_i_absolute: The int index of the head word in the document.

        ne_types: A list of strings denoting all of the types of named
            entities contained within this mention's span.
        ne_spans: A list of int tuples denoting the start and index of the
            contained NE types.
        head_ne_type: A string denoting the type of named entity of this
            mention's head.
        np_type: A string denoting the type of noun phrase of this mention.
        det_type: A string denoting the type of determiner at the start of this
            mention.
        name: A list of strings representing the name of the mention, if the
            mention is a person.

        gender: The gender of the mention, as a string.
        number: The number of the mention, as a string.
        case: A string denoting the grammatical function of the mention.

        synset: The WordNet synset of this mention.
        modifiers: The set of all non-stopword tokens preceding the head word in
            the mention.
        acronym: True if the mention is an acronym, otherwise false.
        sent_id: The number of the sentence in the document.
        
    """

    def __init__(self, start, end, sent, sent_id, word_offset, label=None):
        """Initializes a new Mention object.

        Args:
            start: The int start index of the mention.
            end: The int end index of the mention.
            sent: A list of dictionaries corresponding to rows in a CONLL file.
            sent_id: The number of the sentence in the document.
            word_offset: The total number of words in all previous sentences
                in the document.
            label: The coreference cluster of the mention.
        """
        self.label = label

        head_vals = None
        if end - start == 1:
            head_vals = sent[start]
        else:
            head_vals = self._find_head_values(start, end, sent)
        self.head = head_vals['word']
        self.head_pos = head_vals['pos']
        self.head_lemma = head_vals['lemma']
        self.head_sense = head_vals['sense']
        self.head_i = int(head_vals['word_num'])
        self.head_i_absolute = self.head_i + word_offset

        self.start = start
        self.end = end

        self.ne_types, self.ne_spans = self._find_ne_types_and_spans(start, end, sent)
        self.head_ne_type = self._find_head_ne_type()

        # Readjust some values based on whether or not the mention is a named
        # person.
        self.name = None
        if self.ne_types == ['PERSON'] or self.head_ne_type == 'PERSON':
            self._make_person_adjustments(sent)
        self.np_type = self._find_np_type()
        self.det_type = self._find_det_type(start, sent)

        self.synset = self._find_synset()
        
        self.gender = self._find_gender(sent)
        self.number = self._find_number()

        self.string = ' '.join([vals['word'] for vals in sent[self.start:self.end]])
        self.case = self._find_case(sent)
        if re.match('[A-Z]{2,}', self.head):
            self.acronym = True
        elif re.match('([A-Z].){2,}', self.head):
            self.acronym = True
        else:
            self.acronym = False
        self.modifiers = self._get_modifiers(sent)

        self.sent_id = sent_id


    def _find_head_values(self, start, end, sent):
        """Finds the head of the mention and returns a dictionary of values
        corresponding to the CONLL row of the head word.

        The head-finding algorithm is fairly simple and only relies on the
        parts-of-speech of the words, rather than the parse tree. The algorithm
        either begins at the start of the mention or, if there is a possessive
        marker in the mention, immediately after the possessive marker. It then
        iterates through the words in the mention, stopping when it encounters
        a word that in not a noun, adjective, or cardinal value. It then returns
        the last noun it encountered, or the las word in the mention if it
        did not encounter a noun.

        Args:
            start: The start int index of the mention.
            end: The end int index of the mention.
            sent: A list of dictionaries corresponding to rows in a CONLL file.
        Returns:
            A dictionary corresponding to the CONLL row of the head word.
        """
        pos_list = [vals['pos'] for vals in sent[start:end]]
        start2 = None
        # If a possesive marker occurs in a non-final position, skip all
        # preceding words.
        if 'POS' in pos_list and pos_list[-1] != 'POS':
            start2 = pos_list.index('POS') + 1
        else:
            start2 = start

        # Iterate through the mention's words, storing nouns, skipping CDs and
        # JJs, and stopping on anything else.
        last_noun_vals = None
        for vals in sent[start2:end]:
            pos = vals['pos']
            if pos.startswith('NN'):
                last_noun_vals = vals
            elif pos in ('CD', 'JJ', 'POS'):
                pass
            elif last_noun_vals != None:
                break
        if last_noun_vals == None:
            return sent[end-1]
        else:
            return last_noun_vals

    def _find_np_type(self):
        """Finds the noun phrase type of the mention by utilizing the part of
        speech of the head word. Possible values are 'proper' (for a proper
        noun), 'common' (for a common noun), and 'pronoun' (for a pronoun).

        Returns:
            A string denoting the noun phrase type.
        """
        if self.head_pos in ('NNS', 'NN'):
            return 'common'
        elif self.head_pos == 'NNP':
            return 'proper'
        elif self.head_pos.startswith('PRP') or self.head_pos == 'DT':
            return 'pronoun'
        else:
            return 'unk'

    def _find_det_type(self, start, sent):
        """Finds the type of the determiner of the mention, if it exists.
        Possible values are 'indefinite', 'definite', 'demonstrative', 'unk',
        and 'none'.

        Args:
            start: the starting index of the mention.
            sent: the sentence the mention occurs in.
        Returns:
            The determiner type.
        """
        word = sent[start]['word'].lower()
        pos = sent[start]['pos']
        if pos == 'DT':
            if word in ('a', 'an'):
                return 'indefinite'
            elif word == 'the':
                return 'definite'
            elif word in DEMONSTRATIVES:
                return 'demonstrative'
            else:
                return 'unk'
        return 'none'

    def _find_synset(self):
        """Finds the synset of the of the mention, if it exists.

        Person proper nouns are given the synset 'person.n.01'. The synset of
        all other nouns are looked up using the mention's head. Pronouns are
        assigned a synset based on their gender.

        Returns:
            The synset of the mention, or None if one cannot be found.
        """
        if self.np_type == 'pronoun':
            pronoun_gender = GENDER_DICT.get(self.head.lower(), 'none')
            if pronoun_gender == 'male':
                return MALE_SYNSET
            elif pronoun_gender == 'female':
                return FEMALE_SYNSET
            else:
                return None
        elif self.np_type == 'proper':
            if self.name != None:
                return PERSON_SYNSET
            else:
                synsets = None
                if self.head_lemma != '-':
                    synsets = wordnet.synsets(self.head_lemma, pos='n')
                else:
                    synsets = wordnet.synsets(self.head, pos='n')
        else:
            if self.head_sense != '-':
                synset = None 
                head_synset_name = self.head_lemma + '.n.' + self.head_sense
                try:
                    synset = wordnet.synset(head_synset_name)
                except WordNetError:
                    synset = None
                return synset
            else:
                synsets = None
                if self.head_lemma != '-':
                    synsets = wordnet.synsets(self.head_lemma, pos='n')
                else:
                    synsets = wordnet.synsets(self.head, pos='n')
      
        if synsets:
            return synsets[0]
        else:
            return None

    def _find_gender(self, sent):
        """Determines the gender of the mention. Possible values are 'male',
        'female', 'unk', or 'none'.

        Pronoun genders are drawn from a lookup table, while common nouns check
        to see whether they have the male synset or female synset as their
        hypernym. Person proper nouns use the most likely gender of their
        first name.

        Returns:
            The gender of the mention, or 'none' if it has none.
        """
        if self.np_type == 'pronoun':
            if self.head.lower() in GENDER_DICT:
                return GENDER_DICT[self.head.lower()]
            else:
                return 'none'
        elif self.np_type == 'proper' and self.name != None:
            if self.start != 0:
                prev_word = sent[self.start-1]['word'].lower()
                if prev_word in MALE_HONORIFICS:
                    return 'male'
                elif prev_word in FEMALE_HONORIFICS:
                    return 'female'
                else:
                    return NAME_DICT.get(self.name[0], 'unk')
        else:
            # Determine gender by common hypernyms of the head word and 'male'
            # and 'female'.
            if self.synset is not None:
                def_words = set(self.synset.definition().split()[:4])
                m_hypernyms = self.synset.lowest_common_hypernyms(MALE_SYNSET)
                f_hypernyms = self.synset.lowest_common_hypernyms(FEMALE_SYNSET)
                male_def_match = len(MALE_WORDS.intersection(def_words)) > 0
                female_def_match = len(FEMALE_WORDS.intersection(def_words)) > 0
                if MALE_SYNSET in m_hypernyms or male_def_match:
                    return 'male'
                elif FEMALE_SYNSET in f_hypernyms or female_def_match:
                    return 'female'
                elif PERSON_SYNSET in f_hypernyms:
                    return 'unk'
                else:
                    return 'none'
            else:
                return 'none'

    def _find_number(self):
        """Determines whether the mention is plural of singular. The number of
        pronouns is looked up in a dictionary, while all other mentions check
        the POS of the head word.

        Returns:
            A string indicating whether the mention is singular or plural.
        """
        if self.np_type == 'pronoun':
            if self.head.lower() in PLURAL_PRONOUNS:
                return 'plural'
            else:
                return 'singular'
        elif self.head_pos.startswith('N'):
            if self.head_pos.endswith('S'):
                return 'plural'
            else:
                return 'singular'
        else:
            return 'unk'

    def _find_case(self, sent):
        """Determines the grammatical function of the mention using simple
        heuristics.

        Possible values are:
            'subj' (if the mention is a subject)
            'obj' (if the mention is an object)
            'pos' (if the mention is a possessive)
            'unk' (if the mention's function is unknown)

        Args:
            sent: A list of dictionaries corresponding to rows in a CONLL file.
        Returns:
            A string denoting the grammatical function of the mention.
        """
        # If the head is one of a number of case-specific pronouns:
        if self.np_type == 'pronoun':
            if self.head.lower() in ['I', 'we', 'he', 'she', 'they']:
                return 'subj'
            elif self.head.lower() in ['me', 'us', 'him', 'her', 'them']:
                return 'obj'

        # Use a rough heuristic to determine case.
        if sent[self.end-1]['pos'] == 'POS' or self.head_pos == 'PRP$':
            return 'pos'
        elif self.end != len(sent) and sent[self.end]['parse'].startswith('(VP'):
            return 'subj'
        elif self.start != 0 and sent[self.start-1]['parse'].startswith('(VP'):
            return 'obj'
        elif self.start != 0 and sent[self.start-1]['parse'].startswith('(PP'):
            return 'obj'
        else:
            return 'unk'

    def _find_ne_types_and_spans(self, start, end, sent):
        """Finds all of the named entity types (and their corresponding spans)
        that occur within the mention.

        Args:
            start: The start int index of the mention.
            end: The end int index of the mention.
            sent: A list of dictionaries corresponding to rows in a CONLL file.
        Returns:
            A tuple of a list of the named entity types and a list of int tuples
            corresponding to their spans.
        """
        ne_types = []
        ne_spans = []
        within_flag = False
        for i in range(start, end):
            ne = sent[i]['ne']
            singleton_match = re.match('^\((\w+)\)', ne)
            start_match = re.match('^\((\w+)\*', ne)
            end_match = re.match('^\*\)', ne)
            if singleton_match:
                ne_type = singleton_match.group(1)
                ne_types.append(ne_type)
                ne_spans.append([i, i+1])
            elif start_match:
                ne_type = start_match.group(1)
                ne_types.append(ne_type)
                ne_spans.append([i])
                within_flag = True
            elif end_match and within_flag:
                ne_spans[-1].append(i+1)
                within_flag = False
        for span in ne_spans:
            if len(span) < 2:
                span.append(self.end)
        return ne_types, ne_spans
                
    def _get_modifiers(self, sent):
        """Finds the modifiers if the mention's head word, which are defined as
        all non-stopword words preceding the head word.

        Args:
            sent: A list of dictionaries corresponding to rows in a CONLL file.
        Returns:
            A set of string modifiers.
        """
        modifiers = set()
        for vals in sent[self.start:self.head_i]:
            token = vals['word'].lower()
            if token not in STOPWORDS:
                modifiers.add(token)
        return modifiers

    def _get_person_span(self):
        """If the mention contains a named entity span, returns the span of that
        named entity.
        """
        for ne, span in zip(self.ne_types, self.ne_spans):
            if ne == 'PERSON':
                return span
        return None

    def _find_head_ne_type(self):
        """Finds the named entity type of the head, if it has one.
        """
        for i in range(len(self.ne_types)):
            start, end = self.ne_spans[i]
            if self.head_i >= start and self.head_i < end:
                return self.ne_types[i]
        return 'none'

    def _make_person_adjustments(self, sent):
        """Adjusts several attributes of the mention so it is more accurately
        represented.
        """
        ne_start, ne_end = self._get_person_span()
        head_vals = sent[ne_end-1]
        self.head = head_vals['word']
        self.head_pos = head_vals['pos']
        self.head_lemma = head_vals['lemma']
        self.head_sense = head_vals['sense']
        self.head_i = int(head_vals['word_num'])
        self.name = [vals['word'].lower() for vals in sent[ne_start:ne_end]]


class MentionPair:
    """Objects of this class represent pairs of two different mentions in a
    documents. It possesses methods to extract individual and pairwise features
    from the mention that can later be used for coreference resolution.

    Attributes:
        gender_match: A string denoting whether the two mentions possess the
            same gender.
        number_match: A string denoting whether the two mentions have the same
            number.
        i_np_type: The noun phrase type of mention i, as a string.
        j_np_type: The noun phrase type of mention j, as a string.
        np_trans: The transition of NP types from i to j, as a string.
        i_head_ne: The named entity type of the head of mention i, as a string.
        j_head_ne: The named entity type of the head of mention j, as a string.
        head_ne_trans: The transition of head NE types from i to j, as a string.
        i_case: The grammatical function/case of mention i.
        j_case: The grammatical function/case of mention j.
        case_trans: The transition of case from mention i to mention j, as a
            string.
            
        head_match: A string denoting whether mentions i and j have the same
            head word.
        str_exact_match: A string denoting whether the strings of both mentions
            match exactly.
        str_lower_match: A string denoting whether the lower-cased strings of
            both mentions match.
        mod_match: A string denoting whether mentions i and j have any common
            modifiers.

        sent_distance: The number of sentences between each mentions, as a
            string. Has a value of '0' for mentions in the same sentence, and
            a maximum value of '6'.
        head_distance: The number of words between the heads of each mention, as
            a string.

        common_hypernyms: A string denoting whether mentions i and j have any
            common immediate hypernyms.
        hypernyms_match: A string denoting whether one mention is the hypernym
            of the other.
        synonym_match: A string denoting whether the mentions are synonyms.
        acronym: A string denoting whether one mention is an acronyms of the
            other, or 'N/A' if neither is an acronym.
        embedded: A string denoting whether one mention occurs within the span
            of another.
        
    """

    def __init__(self, i, j):
        """Initializes a new MentionPair object.

        Args:
            i: The antecedent mention, as a Mention object.
            j: The following mention, as a Mention object.
        """

        # Agreement Features
        self.gender_match = self._have_same_gender(i, j)
        self.number_match = self._have_same_number(i, j)

        # Type and Type Transition Features
        self.i_np_type = i.np_type
        self.j_np_type = j.np_type
        self.np_trans = '{}_{}'.format(i.np_type, j.np_type)
        
        #self.i_head_ne = i.head_ne_type
        #self.j_head_ne = j.head_ne_type
        #self.head_ne_trans = '{}_{}'.format(i.head_ne_type, j.head_ne_type)

        self.i_case = i.case
        self.j_case = j.case
        self.case_trans = '{}_{}'.format(i.case, j.case)

        # String Matching Features
        self.head_match = str(i.head.lower() == j.head.lower())
        self.str_exact_match = str(i.string == j.string)
        self.str_lower_match = str(i.string.lower() == j.string.lower())
        self.mod_match = str(len(i.modifiers.intersection(j.modifiers)) > 0)

        # Distance Features
        #self.sent_distance = self._categorize_sentence_distance(i, j)
        #self.head_distance = self._categorize_head_distance(i, j)

        # Wordnet Features
        #self.common_hypernyms = self._have_common_hypernym(i, j)
        #self.hypernym_match = self._one_is_hypernym(i, j)
        #self.synonym_match = self._are_synonyms(i, j)

        # Other Features
        self.acronym = self._one_is_acronym(i, j)
        if i.start <= j.start and i.end >= j.end:
            self.embedded = 'True'
        else:
            self.embedded = 'False'

    def _have_same_gender(self, i, j):
        """Returns 'True' if the given mentions have the same gender, 'False'
        if they do not, and 'Unk' if either mention has an unknown gender.
        Mentions with identical heads are assumed to have the same gender.
        """
        if i.head.lower() == j.head.lower():
            return 'True'
        elif i.gender == 'unk' or j.gender == 'unk':
            return 'Unk'
        else:
            return str(i.gender == j.gender)

    def _have_same_number(self, i, j):
        """Returns 'True' if the given mentions have the same number, 'False'
        if they do not, and 'Unk' if either mention has an unknown number.
        Mentions with identical heads are assumed to have the same number.
        """
        if i.head == j.head:
            return 'True'
        elif i.number == 'unk' or j.number == 'unk':
            return 'Unk'
        else:
            return str(i.number == j.number)

    def _categorize_sentence_distance(self, i, j):
        """Categorizes the sentence distance between the given mentions.

        For distances of 5 or less, the distance (converted to a string) is
        returned. For distance of 6 or greater, a distance of '6' is returned.
        """
        distance = abs(i.sent_id - j.sent_id)
        if distance > 5:
            return '6'
        else:
            return str(distance)

    def _categorize_head_distance(self, i, j):
        """Categorizes the distance between the heads of the given mentions.

        For distances of 5 of below, the distance (converted to a string) is
        returned. For distances from 6 to 10, '10' is returned. For distances
        from 11 to 15, '15' is returned. For distances 16 of above, '20' is
        returned.

        """
        distance = abs(i.head_i_absolute - j.head_i_absolute)
        if distance <= 5:
            return str(distance)
        elif distance <= 10:
            return '10'
        elif distance <= 15:
            return '15'
        else:
            return '20'
            
    def _have_common_hypernym(self, i, j):
        """Returns 'True' if the heads of each mention share a hypernym, 'False'
        if the do not, and 'Unk' if the synset of either mention is unknown.
        """
        if i.synset is not None and j.synset is not None:
            i_hypernyms = set(i.synset.hypernyms())
            j_hypernyms = set(j.synset.hypernyms())
            common_hypernyms = i_hypernyms.intersection(j_hypernyms)
            if len(common_hypernyms - VAGUE_SYNSETS) > 0:
                return 'True'
            else:
                return 'False'
        else:
            return 'Unk'

    def _one_is_hypernym(self, i, j):
        """Returns 'True' if the synset of one mention is a hypernym of the
        other, 'False' if neither are hypernyms of the other, and 'Unk' if the
        synset of wither mention is unknown.
        """
        if i.synset is not None and j.synset is not None:
            if i.synset in j.synset.hypernyms():
                return 'True'
            elif j.synset in i.synset.hypernyms():
                return 'True'
            else:
                return 'False'
        else:
            return 'Unk'

    def _are_synonyms(self, i, j):
        """Returns 'True' if the lemmatized head of one mention occurs in the
        list of lemmas in the other's synset, otherwise returns 'False'.
        """
        if i.synset is not None and j.synset is not None:
            if wordnet.morphy(i.head) in j.synset.lemma_names():
                return 'True'
            elif wordnet.morphy(j.head) in i.synset.lemma_names():
                return 'True'
            else:
                return 'False'
        else:
            return 'False'

    def _one_is_acronym(self, i, j):
        """Returns 'N/A' if neither or both mentions are acronyms, 'True' if one
        is an acronym of the other', and 'False' if one is an acronym, but not
        of the other mention.
        """
        acronym = None
        initial_letters = None
        
        if i.acronym and j.acronym:
            return 'N/A'
        elif not i.acronym and not j.acronym:
            return 'N/A'
        else:
            if i.acronym:
                acronym = i.head.lower()
                initial_letters = ''.join([word[0] for word in j.string.split()])
            elif j.acronym:
                acronym = j.head.lower()
                initial_letters = ''.join([word[0] for word in i.string.split()])

            if acronym in initial_letters:
                return 'True'
            else:
                return 'False'

    def to_dict(self):
        """Returns a dictionary of this objects attributes and their
        corresponding values.
        """
        return vars(self)

