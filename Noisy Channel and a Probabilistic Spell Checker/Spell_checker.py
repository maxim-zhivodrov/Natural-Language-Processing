import math
import random


class Spell_Checker:
    """
    The class implements a context sensitive spell checker. The corrections
    are done in the Noisy Channel framework, based on a language model and
    an error distribution model.
    """

    def __init__(self, lm = None):
        """
        Initializing a spell checker object with a language model as an
        instance  variable.

        Args:
            lm: a language model object. Defaults to None
        """
        self.lm = lm
        self.et = {}
        self.normalization_dict = {'deletion':{}, 'insertion':{}, 'substitution':{}, 'transposition':{}}

    def add_language_model(self, lm):
        """
        Adds the specified language model as an instance variable.
            (Replaces an older LM dictionary if set)

        Args:
            lm: a Spell_Checker.Language_Model object
        """
        self.lm = lm


    def add_error_tables(self, error_tables):
        """
        Adds the speficied dictionary of error tables as an instance variable.
            (Replaces an older value dictionary if set)

        Args:
            error_tables (dict): a dictionary of error tables in the format
            returned by  learn_error_tables()
        """
        self.et = error_tables

    def evaluate(self, text):
        """
        Returns the log-likelihod of the specified text given the language
            model in use. Smoothing should be applied on texts containing OOV words

       Args:
           text (str): Text to evaluate.

       Returns:
           Float. The float should reflect the (log) probability.
           The base of the log operation is 10.
        """
        return self.lm.evaluate(text)

    def spell_check(self, text, alpha):
        """
        Returns the most probable fix for the specified text. Use a simple
        noisy channel model is the number of tokens in the specified text is
        smaller than the length (n) of the language model.

        Args:
            text (str): the text to spell check.
            alpha (float): the probability of keeping a lexical word as is.

        Return:
            A modified string (or a copy of the original if no corrections are made.)
        """
        normalized_text = normalize_text(text)
        str_parts = normalized_text.split()
        wrong_word = self.check_if_has_wrong_word(str_parts)
        if wrong_word is not None:
            if len(str_parts) < self.lm.n:
                fixed_word = self.simple_noisy_chanel(wrong_word)[0]
                fixed_word = str(fixed_word)
                return self.get_fixed_original_sentence(text, wrong_word, fixed_word)
            else:
                fixed_word = self.context_noisy_chanel(str_parts,wrong_word)[0]
                fixed_word = str(fixed_word)
                return self.get_fixed_original_sentence(text, wrong_word, fixed_word)
        else:
            if len(str_parts) < self.lm.n:
                fixed_word, wrong_word = self.simple_real_words_noisy_chanel(str_parts, alpha)
                fixed_word, wrong_word = str(fixed_word), str(wrong_word)
                return self.get_fixed_original_sentence(text, wrong_word, fixed_word)
            else:
                fixed_word, wrong_word = self.context_real_words_noisy_chanel(str_parts, alpha)
                fixed_word, wrong_word = str(fixed_word), str(wrong_word)
                return self.get_fixed_original_sentence(text, wrong_word, fixed_word)



    # region My method spelling checker
    # region Get edit candidates
    def get_candidates(self, word):
        """
        Returns the candidates of the specific word up to 2 edit
        Args:
            word (str): the word to find candidates to
        Returns:
                (tuple): two lists of candidates
        """
        one_edit_candidates = self.get_edits_by_one(word=word)
        two_edit_candidates = self.get_edits_by_two(self.get_edits_by_one(word = word, two_edits_first_round=True), word)
        return one_edit_candidates, two_edit_candidates

    def get_edits_by_one(self, word, two_edits_first_round = False, original_word = None):
        """
        Returns all candidates that one edit away from word
        Args:
            word (str): the word to find candidates for
            two_edits_first_round (bool): know if clean from known words
            original_word (str): the original word before the changes
        Returns:
            edits_dict (dict): one edit candidates and the changed letters
        """
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        insertion = self.get_insertion(splits, letters)
        transposition = self.get_transposition(splits, letters)
        substitution = self.get_substitution(splits, letters)
        deletion = self.get_deletion(splits, letters)

        if not two_edits_first_round: # For not removing if two edits needed
            if original_word is not None: word = original_word
            deletion = self.remove_redundant_words(deletion, word)
            transposition = self.remove_redundant_words(transposition, word)
            substitution = self.remove_redundant_words(substitution, word)
            insertion = self.remove_redundant_words(insertion, word)


        edits_dict = {'deletion':set(deletion),
                      'transposition':set(transposition),
                      'substitution':set(substitution),
                      'insertion':set(insertion)}
        return edits_dict

    def get_edits_by_two(self, one_edit_candidates, original_word):
        """
        Returns the two edit candidates for the specific word
        Args:
            one_edit_candidates (dict): the candidates after one edit
            original_word (str): the original word that found candidates for
        Returns:
            two_edit_candidates (dict): the two edit candidates and the letter changed
        """
        error_types = one_edit_candidates.keys()
        two_edit_candidates = {f'{error_one}+{error_two}': set() for error_one in error_types for error_two in error_types}

        for error_one, candidate_set_one in one_edit_candidates.items():
            for tpl_one in candidate_set_one:
                candidate_one = tpl_one[0]
                two_letters_one = tpl_one[1]
                second_edits = self.get_edits_by_one(candidate_one, original_word = original_word)
                for error_two, candidate_set_two in second_edits.items():
                    for tpl_two in candidate_set_two:
                        candidate_two = tpl_two[0]
                        two_letters_two = tpl_two[1]
                        two_edit_candidates[f'{error_one}+{error_two}'].add((candidate_two, f'{two_letters_one}+{two_letters_two}'))
        return two_edit_candidates

    def get_deletion(self, splits, letters):
        """
        Returns the deletion edits for word
        Args:
             splits (list): list of all splits for the word
             letter (str): all english letters
        Returns:
                (list): all deletion edits
        """
        deletion = []
        for L, R in splits:
            for c in letters:
                if L != '':
                    deletion.append((L + c + R, L[-1] + c))
                else:
                    deletion.append((L + c + R, '#' + c))
        return deletion

    def get_insertion(self, splits, letters):
        """
        Returns the insertion edits for word
        Args:
             splits (list): list of all splits for the word
             letter (str): all english letters
        Returns:
                (list): all insertion edits
        """
        insertion = []
        for L, R in splits:
            if R:
                if L != '':
                    insertion.append((L + R[1:], L[-1] + R[0]))
                else:
                    insertion.append((L + R[1:], '#' + R[0]))
        return insertion

    def get_transposition(self, splits, letters):
        """
        Returns the transposition edits for word
        Args:
             splits (list): list of all splits for the word
             letter (str): all english letters
        Returns:
                (list): all transposition edits
        """
        transposition = []
        for L, R in splits:
            if len(R) > 1:
                if R[0] == R[1]: continue
                transposition.append((L + R[1] + R[0] + R[2:], R[1] + R[0]))
        return transposition

    def get_substitution(self, splits, letters):
        """
        Returns the substitution edits for word
        Args:
             splits (list): list of all splits for the word
             letter (str): all english letters
        Returns:
                (list): all substitution edits
        """
        substitution = []
        for L, R in splits:
            if R:
                for c in letters:
                    if c == R[0]: continue
                    substitution.append((L + c + R[1:], R[0]+c))
        return substitution
    # endregion

    # region Normzalization methods
    def deletion_normalization(self, string):
        """
        Returns the normalization for deletion edit for two letters
        Args:
            string (str): two letters
        Returns:
                (int): normalization factor for deletion
        """
        if string in self.normalization_dict['deletion']:
            return self.normalization_dict['deletion'][string]
        if string[0] != '#':
            count_appearences = sum([k.count(string) for k in self.lm.WORDS.keys()])
        else:
            count_appearences = len([1 for k in self.lm.WORDS.keys() if k[0] == string[1]])
        self.normalization_dict['deletion'][string] = count_appearences
        return count_appearences

    def insertion_normalization(self, string):
        """
        Returns the normalization for insertion edit for two letters
        Args:
            string (str): two letters
        Returns:
                (int): normalization factor for insertion
        """
        if string in self.normalization_dict['insertion']:
            return self.normalization_dict['insertion'][string]
        if string[0] != '#':
            count_appearences = sum([k.count(string[0]) for k in self.lm.WORDS.keys()])
        else:
            count_appearences = len(self.lm.WORDS)
        self.normalization_dict['insertion'][string] = count_appearences
        return count_appearences

    def substitution_normalization(self, string):
        """
        Returns the normalization for substitution edit for two letters
        Args:
            string (str): two letters
        Returns:
                (int): normalization factor for substitution
        """
        if string in self.normalization_dict['substitution']:
            return self.normalization_dict['substitution'][string]
        count_appearences = sum([k.count(string[1]) for k in self.lm.WORDS.keys()])
        self.normalization_dict['substitution'][string] = count_appearences
        return count_appearences

    def transposition_normalization(self, string):
        """
        Returns the normalization for transposition edit for two letters
        Args:
            string (str): two letters
        Returns:
                (int): normalization factor for transposition
        """
        if string in self.normalization_dict['transposition']:
            return self.normalization_dict['transposition'][string]
        count_appearences = sum([k.count(string) for k in self.lm.WORDS.keys()])
        self.normalization_dict['transposition'][string] = count_appearences
        return count_appearences
    # endregion

    def simple_noisy_chanel(self, word):
        """
        Returns the correction of the word if simple noisy chanel applied
        Args:
            word (str): word to fix
        Returns:
                (tuple): fixed word, probability
        """
        candidates_mistake_probs = self.calculate_mistake_prob(word)
        candidates_chanel_probs = []
        for candidate_mistake_tpl in candidates_mistake_probs:
            candidate_word = candidate_mistake_tpl[0]
            mistake_prob = candidate_mistake_tpl[1]
            candidates_chanel_probs.append((candidate_word,mistake_prob*self.calculate_prior_prob(candidate_word)))

        max_prob_tpl = self.get_tuple_with_max_values(candidates_chanel_probs)
        return max_prob_tpl

    def context_noisy_chanel(self,str_parts ,word):
        """
        Returns the correction of the word if context noisy chanel applied
        Args:
            str_parts (list): the sentence seperated to tokens
            word (str): word to fix
        Returns:
                (tuple): fixed word, probability
        """
        all_grams_containing_word = self.get_all_grams_contains_the_word(str_parts, word)
        candidates_mistakes_probs  = self.calculate_mistake_prob(word)
        candidates_chanel_probs = []

        for candidate_mistake_tpl in candidates_mistakes_probs:
            candidate_word, mistake_prob = candidate_mistake_tpl
            all_grams_containing_candidate_word = [gram.replace(word, candidate_word) for gram in all_grams_containing_word]
            gram_probabilty = 1
            for gram in all_grams_containing_candidate_word:
                gram_probabilty *= math.pow(10, self.evaluate(gram))
            candidates_chanel_probs.append((candidate_word, mistake_prob*gram_probabilty))

        max_prob_tpl = self.get_tuple_with_max_values(candidates_chanel_probs)
        return max_prob_tpl

    def simple_real_words_noisy_chanel(self, str_parts, alpha):
        """
        Returns the correction of the word if simple real words noisy chanel applied
        Args:
            str_parts (list): the sentence sapareted to tokens
            alpha (double): probability for the word not to be a wrong word
        Returns:
                (tuple): fixed word, the wrong word
        """
        candidates_for_real_mistake = {}
        for word in str_parts:
            best_simple_candidate_word = self.simple_noisy_chanel(word)
            prior_probability = self.calculate_prior_prob(word)
            if alpha * prior_probability > best_simple_candidate_word[1]:
                candidates_for_real_mistake[(word, prior_probability)] = word
            else:
                candidates_for_real_mistake[best_simple_candidate_word] = word

        temp_candidate_dict = {tpl: original_word for tpl, original_word in candidates_for_real_mistake.items() if tpl[0] != original_word}
        if len(temp_candidate_dict) != 0:
            candidates_for_real_mistake = temp_candidate_dict

        max_prob_tpl = self.get_tuple_with_max_values(list(candidates_for_real_mistake.keys()))
        return max_prob_tpl[0], candidates_for_real_mistake[max_prob_tpl]

    def context_real_words_noisy_chanel(self, str_parts, alpha):
        """
        Returns the correction of the word if context real words noisy chanel applied
        Args:
            str_parts (list): the sentence sapareted to tokens
            alpha (double): probability for the word not to be a wrong word
        Returns:
                (tuple): fixed word, the wrong word
        """
        candidates_for_real_mistake = {} # Key: Tuple of (candidate_word, probability), Value: Original word
        for word in str_parts:
            best_context_candidate_tpl = self.context_noisy_chanel(str_parts, word)
            all_grams_containing_word = self.get_all_grams_contains_the_word(str_parts, word)
            gram_probability = 1
            for gram in all_grams_containing_word:
                gram_probability *= math.pow(10, self.evaluate(gram))

            if alpha * gram_probability > best_context_candidate_tpl[1]:
                candidates_for_real_mistake[(word, alpha * gram_probability)] = word
            else:
                candidates_for_real_mistake[best_context_candidate_tpl] = word

        temp_candidate_dict = {tpl:original_word for tpl, original_word in candidates_for_real_mistake.items() if tpl[0] != original_word}
        if len(temp_candidate_dict) != 0:
            candidates_for_real_mistake = temp_candidate_dict

        max_prob_tpl = self.get_tuple_with_max_values(list(candidates_for_real_mistake.keys()))
        return max_prob_tpl[0], candidates_for_real_mistake[max_prob_tpl]

    def calculate_prior_prob(self, word):
        """
        Returns the prior probability of the word
        Args:
            word (str): word to find prior probability
        Returns:
                (double): prior probability
        """
        WORDS = self.lm.WORDS
        N = sum(WORDS.values())
        return (WORDS[word]) / N

    def calculate_mistake_prob(self, word):
        """
        Returns the noisy chanel probability of thr word
        Args:
             word (str): word to find noisy chanel probability
        Returns:
                (double): noisy chanel probability
        """
        candidates_mistake_probs = [] # Tuples of (candidate_word, mistake_probabilty)
        one_edit_candidates, two_edit_candidates = self.get_candidates(word)
        norm_methods = {'deletion':self.deletion_normalization, 'insertion': self.insertion_normalization,
                        'substitution':self.substitution_normalization, 'transposition':self.transposition_normalization}

        for error_type, candidates_letters_tpls in one_edit_candidates.items():
            for can_let_tpl in candidates_letters_tpls:
                candidate_word = can_let_tpl[0]
                letters_modified = can_let_tpl[1]
                try:
                    error_model_prob = self.et[error_type][letters_modified]
                    if error_model_prob == 0: continue
                    mistake_prob = error_model_prob / (norm_methods[error_type](letters_modified))
                except (ZeroDivisionError, KeyError): mistake_prob = 0
                if mistake_prob != 0:
                    candidates_mistake_probs.append((candidate_word, mistake_prob))

        for error_type, candidates_letters_tpls in two_edit_candidates.items():
            first_error, second_error = error_type.split('+')[0], error_type.split('+')[1]
            for can_let_tpl in candidates_letters_tpls:
                candidate_word = can_let_tpl[0]
                letters_modified = can_let_tpl[1]
                first_letters_modified, second_letters_modified = letters_modified.split('+')[0], letters_modified.split('+')[1]
                try:
                    error_model_prob_one = self.et[first_error][first_letters_modified]
                    error_model_prob_two = self.et[second_error][second_letters_modified]
                    if error_model_prob_one == 0 or error_model_prob_two == 0: continue
                    first_mistake_prob = error_model_prob_one / (norm_methods[first_error](first_letters_modified))
                    second_mistake_prob = error_model_prob_two / (norm_methods[second_error](second_letters_modified))
                    mistake_prob = first_mistake_prob * second_mistake_prob
                except (ZeroDivisionError, KeyError): mistake_prob = 0
                if mistake_prob != 0:
                    candidates_mistake_probs.append((candidate_word, mistake_prob))

        return candidates_mistake_probs

    # region Helpful methods
    def get_all_grams_contains_the_word(self, str_parts, word):
        """
        Returns all grams containing the word
        Args:
            str_parts (list): the sentences seperated to tokens
            word (str): the word to find grams with
        Returns:
                (list): list of grams
        """
        all_grams = []
        bound = len(str_parts) - self.lm.n + 1
        N = self.lm.n
        for i in range(bound):
            curr_gram = ' '.join(str_parts[i:i + N])
            all_grams.append(curr_gram)
        return [gram for gram in all_grams if word in gram.split()]

    def get_tuple_with_max_values(self, lst):
        """
        Returns tuple with maximum value ([1] cell) from list
        Args:
             lst (list): list of tuples
        Returns:
                (tuple): tuple with maximum value
        """
        max_value_tpl = (0, 0)
        for tpl in lst:
            if tpl[1] > max_value_tpl[1]:
                max_value_tpl = tpl
        return max_value_tpl

    def check_if_has_wrong_word(self, str_parts):
        """
        Checks whether the sentence has wrong word
        Args:
            str_parts (list): the sentence seperated to tokens
        Returns:
                (bool): indicator if the sentence has a wrong word
        """
        WORDS = self.lm.WORDS
        for word in str_parts:
            if word not in WORDS:
                return word
        return None

    def remove_redundant_words(self, lst, original_word):
        """
        Removes words that are not in the vocabluary
        Args:
             lst (list): list of edit candidates
             original_word (str): the original word
        Returns:
                (list): candidate list without redundant words
        """
        WORDS = self.lm.WORDS
        return [tpl for tpl in lst if tpl[0] in WORDS and tpl[0] != original_word]


    def get_fixed_original_sentence(self, original_text_wrong, wrong_word, fixed_word):
        """
        Returns the sentence to its previous form after fixation
        Args:
            original_text_wrong (str): original wrong sentence
            wrong_word (str): the wrong word
            fixed_word (str): the fixed word
        Returns:
                (str): fixed sentence with previoud form
        """
        set_punctuations = {char for char in '''!()-[]{};:'"\,<>./?@#$%^&*_~'''}

        # No punctuations and no capital letters
        num_of_punctuations = len([1 for char in original_text_wrong if char in set_punctuations])
        num_of_capital_letters = len([1 for char in original_text_wrong if char.isupper()])
        if num_of_punctuations == 0 and num_of_capital_letters == 0:
            return ' '.join([part if part != wrong_word else fixed_word for part in original_text_wrong.split()])

        # Has at least one punctuation or capital letter
        new_str_parts = []
        str_parts = original_text_wrong.split()
        for part in str_parts:
            if len(part) == 0: continue
            normalized_part = normalize_text(part).replace(' ','')
            if normalized_part != wrong_word:
                new_str_parts.append(part)
                continue
            punctuation_indices = []
            if part[0] in set_punctuations: punctuation_indices.append(0)
            if part[-1] in set_punctuations: punctuation_indices.append(len(part) - 1)

            if 0 in  punctuation_indices and len(part) == 1:continue # if only punctuation

            has_capital_letter = False
            if 0 in punctuation_indices: has_capital_letter = part[1].isupper()
            else: has_capital_letter = part[0].isupper()

            new_part = fixed_word

            try:
                if has_capital_letter: new_part = new_part.capitalize()
            except AttributeError:
                x = 0
            for index in punctuation_indices:
                if index == 0: new_part = part[index] + new_part
                elif index == len(part) - 1: new_part = new_part + part[index]

            new_str_parts.append(new_part)

        return ' '.join(new_str_parts)


    # endregion

    # endregion


    # region Language model inner class
    class Language_Model:
        """
        The class implements a Markov Language Model that learns amodel from a given text.
            It supoprts language generation and the evaluation of a given string.
            The class can be applied on both word level and caracter level.
        """

        def __init__(self, n=3, chars = False):
            """
            Initializing a language model object.
            Args:
                n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
                chars (bool): True iff the model consists of ngrams of characters rather then word tokens.
                              Defaults to False
            """
            self.n = n
            self.chars = chars
            self.model_dict = None # key: N gram, value: how much appeared
            # For smoothing computition
            self.model_trimmed_dict = {} # key: N-1 gram, value: how much appeared
            # For generating text from model
            self.model_context_dict = {} # key N-1 gram, value: list of N grams that N-1 gram is their prefix
            # For spelling correction
            self.WORDS = None

        def build_model(self, text):
            """
            Populates the instance variable model_dict.
            Args:
                text (str): the text to construct the model from.
            """
            normalized_text = normalize_text(text)
            self.model_dict = {}
            if not self.chars:
                str_parts = normalized_text.split()
            else:
                str_parts = [char for char in normalized_text]

            self.WORDS = self.build_word_vocabulary(str_parts) if not self.chars else self.build_word_vocabulary(normalized_text.split())

            dict_lst = [self.model_dict, self.model_trimmed_dict]
            bound_lst = [len(str_parts) - self.n + 1, len(str_parts) - self.n + 2]

            for dct, bound, N in zip(dict_lst, bound_lst, [self.n, self.n - 1]):
                for i in range(bound):  # N-grams
                    curr_gram = ' '.join(str_parts[i:i+N])
                    if curr_gram in dct:
                        dct[curr_gram] += 1
                    else:
                        dct[curr_gram] = 1
                    if dct is self.model_dict:
                        trimmed_curr_gram = curr_gram[0:curr_gram.rindex(' ')]
                        trimmed_part = curr_gram[curr_gram.rindex(' ')+1:]
                        if trimmed_curr_gram in self.model_context_dict:
                            self.model_context_dict[trimmed_curr_gram].append(trimmed_part)
                        else:
                            gram_lst = [trimmed_part]
                            self.model_context_dict[trimmed_curr_gram]=gram_lst


        def get_model_dictionary(self):
            """
            Returns the dictionary class object
            """
            return self.model_dict

        def get_model_window_size(self):
            """
            Returning the size of the context window (the n in "n-gram")
            """
            return self.n

        def generate(self, context=None, n=20):
            """
            Returns a string of the specified length, generated by applying the language model
            to the specified seed context. If no context is specified the context should be sampled
            from the models' contexts distribution. Generation should stop before the n'th word if the
            contexts are exhausted. If the length of the specified context exceeds (or equal to)
            the specified n, the method should return the a prefix of length n of the specified context.

            Args:
                context (str): a seed context to start the generated string from. Defaults to None
                n (int): the length of the string to be generated.

            Return:
                String. The generated text.
            """
            # What to do if the context is not in the dictionary? !!!
            if context is not None and len(context.split()) >= n:
                return (context.split())[0:n]
            if context is None:
                # if sampling is by uniform
                # context = self.choose_sample('uniform')

                # if sampling is by choise
                context = self.choose_sample('choise')

            if not self.chars:
                generated_text = context.split()
            else:
                generated_text = [char for char in context]

            while len(generated_text) < n:
                last_context = generated_text[-(self.n-1):]
                try:
                    last_context_list = self.model_context_dict[' '.join(last_context)]
                    random_number = random.randint(0, len(last_context_list)-1)
                    generated_text.append(last_context_list[random_number])
                except KeyError: break # Stops when context in not in the text
            return ' '.join(generated_text)

        def evaluate(self, text):
            """
            Returns the log-likelihood of the specified text to be a product of the model.
               Laplace smoothing should be applied if necessary.

            Args:
                text (str): Text to evaluate.

            Returns:
                Float. The float should reflect the (log) probability.
                The base of the log operation is 10.
            """
            if not self.chars:
                str_parts = text.split()
            else:
                str_parts = [char for char in text]
            bound = len(str_parts) - self.n + 1
            N = self.n
            prob = 1
            for i in range(bound):
                curr_gram = ' '.join(str_parts[i:i+N])
                trimmed_curr_gram = curr_gram[0:curr_gram.rindex(' ')]
                if curr_gram in self.model_dict and trimmed_curr_gram in self.model_trimmed_dict:
                    prob *= self.model_dict[curr_gram] / self.model_trimmed_dict[trimmed_curr_gram]
                else:
                    prob *= self.smooth(curr_gram)
            return math.log(prob, 10)


        def smooth(self, ngram):
            """
            Returns the smoothed (Laplace) probability of the specified ngram.

            Args:
                ngram (str): the ngram to have it's probability smoothed

            Returns:
                float. The smoothed probability.
            """
            trimmed_ngram = ngram[0:ngram.rindex(' ')]
            V = len(self.model_trimmed_dict)
            upper_c = 0 if ngram not in self.model_dict else self.model_dict[ngram]
            lower_c = 0 if trimmed_ngram not in self.model_trimmed_dict else self.model_trimmed_dict[trimmed_ngram]
            return (upper_c+1) / (lower_c+V)

        # My methods
        def choose_sample(self, how):
            """
            Returns the context that has been chosen
            Args:
                how (str): the choosing method
            Returns:
                    (str): the context
            """
            if how == 'uniform':
                context_keys = list(self.model_context_dict.keys())
                random_number = random.randint(0,len(context_keys) - 1)
                return context_keys[random_number]
            elif how == 'choise':
                context_keys, weights = [], []
                for key, value in self.model_context_dict.items():
                    context_keys.append(key)
                    weights.append(len(value))
                return (random.choices(context_keys, weights, k=1))[0]

        def build_word_vocabulary(self, str_parts):
            """
            Builds the word vocabulary
            Args:
                str_parts (list): the text separated to tokens
            Returns:
                    (dict): word vocabulary
            """
            word_dict = {}
            for word in str_parts:
                if word in word_dict:
                    word_dict[word] +=1
                else:
                    word_dict[word] = 1
            return word_dict
    # endregion



# region Normalization and WhoAmI methods

def normalize_text(text):
    """
    First part of normalization: transform the text to lowercase
    Reason behind: to keep the uniformity between all words and sentences, because uppercase words doesn't have
                    much effect on the context of the sentence.

    Second part of normalization: remove punctuation
    Reason behind: remove unnecessary tokens that do not effect on the correction of the sentence

    Third part of normalization: tokenizing
    Reason behind: to separate sentences by whitespace and transform each word into a token that I can work with

    Fourth part of normalization: handle cases like "famous,actress" when no white space separate between two words
    Reason behind: to separate the sentence to number of tokens and not work with bad and useless tokens

    Note: I did not do stemming and lemmatization because I think the suffix of the word effects the right
            correction of the sentence


    Returns a normalized version of the specified string.
    You can add default parameters as you like (they should have default values!)
    You should explain your decisions in the header of the function.

    Args:
        text (str): the text to normalize

    Returns:
        string. the normalized text.
    """
    separated_text = ' '.join(text.split('<s>'))
    lowered_text = separated_text.lower()
    set_punctuations = {char for char in '''!()-[]{};:'"\,<>./?@#$%^&*_~'''}
    set_punctuations.add('\n')
    normalized_text = ''
    for index in range(len(lowered_text)):
        try:
            if lowered_text[index] not in set_punctuations:
                normalized_text += lowered_text[index]
            else:
                try:
                    if lowered_text[index+1] != ' ' and lowered_text[index-1] != ' ':
                        normalized_text += ' '
                except IndexError:
                    continue
        except UnicodeDecodeError:
            continue
    return normalized_text


def who_am_i():
    """
    Returns a ductionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Maxim Zhivodrov', 'id': '317649606', 'email': 'maximzh@post.bgu.ac.il'}
# endregion





