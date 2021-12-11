"""
intro2nlp, assignment 3, 2021

In this assignment you will implement a Hidden Markov model
to predict the part of speech sequence for a given sentence.

"""


from math import log, isfinite, pow
from collections import Counter

import sys, os, time, platform, nltk, random


def main():
    """
    Main method used for tests
    """
    x = 0
    # global START,END, UNK, allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B, num_of_sentences
    # train_sentences = load_annotated_corpus('en-ud-train.upos.tsv')
    # learn_params(train_sentences)
    # test_sentences = load_annotated_corpus('en-ud-dev.upos.tsv')
    #
    #
    # predicted_sentences = [tag_sentence([word for word,tag in sentence], {'baseline':[perWordTagCounts,allTagCounts]})
    #                   for sentence in test_sentences]
    # right_predictions = 0
    # for gold_sentence, pred_sentence in zip(test_sentences, predicted_sentences):
    #     right_predictions += count_correct(gold_sentence, pred_sentence)[0]
    #
    # predicted_sentences = [tag_sentence([word for word,tag in sentence], {'hmm':[A,B]}) for sentence in test_sentences]
    # right_predictions = 0
    # for gold_sentence, pred_sentence in zip(test_sentences, predicted_sentences):
    #     right_predictions += count_correct(gold_sentence, pred_sentence)[0]


# utility functions to read the corpus
def who_am_i(): #this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    #TODO edit the dictionary to have your own details
    return {'name': 'Maxim Zhivodrov', 'id': '317649606', 'email': 'maximzh@post.bgu.ac.il'}


def read_annotated_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append( (word, tag) )
        line = f.readline()
    return sentence

def load_annotated_corpus(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_annotated_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_annotated_sentence(f)
    return sentences


START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"
allTagCounts = Counter()
# use Counters inside these
perWordTagCounts = {}
emissionCounts = {}
transitionCounts = {}
# log probability distributions: do NOT use Counters inside these because
# missing Counter entries default to 0, not log(0)
A = {} #transisions probabilities
B = {} #emmissions probabilities
num_of_sentences = 0

def learn_params(tagged_sentences):
    """
    Populates and returns the allTagCounts, perWordTagCounts, transitionCounts,
       and emissionCounts data-structures.
      allTagCounts and perWordTagCounts should be used for baseline tagging and
      should not include pseudocounts, dummy tags and unknowns.
      The transisionCounts and emmisionCounts
      should be computed with pseudo tags and shoud be smoothed.
      A and B should be the log-probability of the normalized counts, based on
      transisionCounts and  emmisionCounts


    Args:
        tagged_sentences: a list of tagged sentences, each tagged sentence is a
        list of pairs (w,t), as retunred by load_annotated_corpus().

    Return:
        [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] (a list)
    """
    #TODO complete the code

    """
    When an unknown transition between two tags or 
    an unknown emission from a tag to word appears a
    Laplace smoothing applied.
    """

    global START,END, UNK, allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B, num_of_sentences

    all_tags = []
    all_words = []
    for sentence in tagged_sentences:
        for word, tag in sentence:
            all_words.append(word)
            all_tags.append(tag)

    set_all_words = set(all_words)
    set_all_tags = set(all_tags)
    num_of_sentences = len(tagged_sentences)

    dict_end_tag = {tag:0 for tag in set_all_tags} #How much each tag appeared in the end of sentence
    for sentence in tagged_sentences:
        last_tag = sentence[-1][1]
        dict_end_tag[last_tag] += 1

    #allTagsCount population
    allTagCounts.update(all_tags)

    #perWordTagCounts population
    for word in all_words:
        perWordTagCounts[word] = {tag:0 for tag in set_all_tags}
    for sentence in tagged_sentences:
        for word, tag in sentence:
            perWordTagCounts[word][tag] += 1
    V = len(perWordTagCounts)

    #transitionCounts population
    for tag1 in set_all_tags:
        transitionCounts[(START, tag1)] = 0
        transitionCounts[(tag1, END)] = 0
        for tag2 in set_all_tags:
            transitionCounts[(tag1, tag2)] = 0

    for sentence in tagged_sentences:
        first_tag, last_tag = sentence[0][1], sentence[-1][1]
        transitionCounts[(START, first_tag)] += 1
        transitionCounts[(last_tag, END)] += 1
        for index in range(len(sentence)-1):
            tag_tag_combo = (sentence[index][1], sentence[index+1][1])
            transitionCounts[tag_tag_combo] += 1

    #emissionCounts population
    for tag in set_all_tags:
        for word in set_all_words:
            emissionCounts[(tag, word)] = 0
    for sentence in tagged_sentences:
        for word, tag in sentence:
            emissionCounts[(tag, word)] += 1

    #A population
    for tag_tag_combo, count in transitionCounts.items():
        first_tag,second_tag = tag_tag_combo
        if first_tag == START or second_tag == END:
            if count != 0:
                A[tag_tag_combo] = log(count / num_of_sentences, 10)
            else:
                A[tag_tag_combo] = log(1 / (num_of_sentences + V), 10)
        else:
            if count != 0:
                A[tag_tag_combo] = log(count / allTagCounts[first_tag], 10)
            else:
                A[tag_tag_combo] = log(1 / (allTagCounts[first_tag] + V), 10)

    #B population
    for tag_word_combo, count in emissionCounts.items():
        tag, word = tag_word_combo
        if count != 0:
            B[tag_word_combo] = log(count / allTagCounts[tag], 10)
        else:
            B[tag_word_combo] = log(1 / (allTagCounts[tag] + V), 10)

    return [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B]

def baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Each word is tagged by the tag most
    frequently associated with it. OOV words are tagged by sampling from the
    distribution of all tags.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        Return:
        list: list of pairs
    """

    #TODO complete the code
    global START, END, UNK, transitionCounts, emissionCounts, A, B, num_of_sentences
    tagged_sentence = []
    for word in sentence:
        if word in perWordTagCounts:
            max_tag = max(perWordTagCounts[word], key=perWordTagCounts[word].get)
            tagged_sentence.append((word, max_tag))
        else:
            sample = random.choices(list(allTagCounts.keys()), weights = list(allTagCounts.values()), k = 1)[0]
            tagged_sentence.append((word,sample))

    return tagged_sentence

#===========================================
#       POS tagging with HMM
#===========================================


def hmm_tag_sentence(sentence, A, B):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Tagging is done with the Viterby
    algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

    Return:
        list: list of pairs
    """

    #TODO complete the code
    global START, END, UNK, allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, num_of_sentences
    v_last = viterbi(sentence, A, B)
    list_of_tags = retrace(v_last)
    tagged_sentence = [(word,tag) for word, tag in zip(sentence, list_of_tags)]
    return tagged_sentence

def viterbi(sentence, A,B):
    """Creates the Viterbi matrix, column by column. Each column is a list of
    tuples representing cells. Each cell ("item") is a tupple (t,r,p), were
    t is the tag being scored at the current position,
    r is a reference to the corresponding best item from the previous position,
    and p is a log probabilityof the sequence so far).

    The function returns the END item, from which it is possible to
    trace back to the beginning of the sentence.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

    Return:
        obj: the last item, tagged with END. should allow backtraking.

        """
        # Hint 1: For efficiency reasons - for words seen in training there is no
        #      need to consider all tags in the tagset, but only tags seen with that
        #      word. For OOV you have to consider all tags.
        # Hint 2: start with a dummy item  with the START tag (what would it log-prob be?).
        #         current list = [ the dummy item ]
        # Hint 3: end the sequence with a dummy: the highest-scoring item with the tag END


    #TODO complete the code
    """
    When an unknown transition between two tags or 
    an unknown emission from a tag to word appears a
    Laplace smoothing applied.
    """

    global START,END, UNK, allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, num_of_sentences
    set_all_tags = set((dict(allTagCounts)).keys())
    V = len(perWordTagCounts)

    #First iteration
    first_word = sentence[0]
    first_column = []
    for tag in set_all_tags:
        tag_word_combo = (tag, first_word)
        if first_word in perWordTagCounts:
            if tag_word_combo not in emissionCounts: continue
            elif emissionCounts[tag_word_combo]==0: continue
        try:
            A_prob = pow(10, A[(START, tag)])
        except KeyError:
            A_prob = 1/(num_of_sentences + V)
        try:
            B_prob = pow(10, B[tag_word_combo])
        except KeyError:
            tag_counts = 0 if tag not in allTagCounts else allTagCounts[tag]
            B_prob = 1/(tag_counts + V)
        prob = log(A_prob * B_prob, 10)
        first_column.append((tag, START, prob))

    #Other iterations
    curr_column = first_column
    for word in sentence[1:]:
        next_column = []
        for tag in set_all_tags:
            tag_word_combo = (tag, word)
            if word in perWordTagCounts:
                if tag_word_combo not in emissionCounts: continue
                elif emissionCounts[tag_word_combo]==0: continue
            probs_dict = {}
            for column_tag_tpl in curr_column:
                column_tag, previous, column_prob = column_tag_tpl
                try:
                    A_prob = pow(10, A[(column_tag, tag)])
                except KeyError:
                    column_tag_counts = 0 if column_tag not in allTagCounts else allTagCounts[column_tag]
                    A_prob = 1/(column_tag_counts + V)
                try:
                    B_prob = pow(10, B[tag_word_combo])
                except KeyError:
                    tag_counts = 0 if tag not in allTagCounts else allTagCounts[tag]
                    B_prob = 1/(tag_counts + V)
                prob = log(pow(10, column_prob) * A_prob * B_prob, 10)
                probs_dict[column_tag_tpl] = prob
            max_prob_tpl = max(probs_dict, key=probs_dict.get)
            next_column.append((tag, max_prob_tpl, probs_dict[max_prob_tpl]))
        curr_column = next_column


    #Last iteration
    prob_dict = {}
    for column_tag, previous, column_prob in curr_column:
        try:
            A_prob = pow(10, A[(column_tag, END)])
        except KeyError:
            A_prob = 1 / (num_of_sentences + V)
        prob_dict[(column_tag, previous)] = log(pow(10, column_prob) * A_prob, 10)
    max_prob_tpl = max(prob_dict, key = prob_dict.get)
    v_last = (END, max_prob_tpl, prob_dict[max_prob_tpl])

    return v_last

#a suggestion for a helper function. Not an API requirement
def retrace(end_item):
    """Returns a list of tags (retracing the sequence with the highest probability,
        reversing it and returning the list). The list should correspond to the
        list of words in the sentence (same indices).
    """
    global START,END, UNK, allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B, num_of_sentences
    list_of_tags = []
    curr_item = end_item[1]
    while curr_item[1] != START:
        list_of_tags.append(curr_item[0])
        curr_item = curr_item[1]
    list_of_tags.append(curr_item[0])
    list_of_tags.reverse()
    return list_of_tags

#a suggestion for a helper function. Not an API requirement
def predict_next_best(word, tag, predecessor_list):
    """
    Returns a new item (tupple)
    """
    pass



def joint_prob(sentence, A, B):
    """Returns the joint probability of the given sequence of words and tags under
     the HMM model.

     Args:
         sentence (pair): a sequence of pairs (w,t) to compute.
         A (dict): The HMM Transition probabilities
         B (dict): the HMM emmission probabilities.
     """
    p = 1   # joint log prob. of words and tags

    #TODO complete the code
    global START,END, UNK, allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, num_of_sentences
    V = len(perWordTagCounts)
    for i in range(len(sentence)):
        if i == 0:
            word, tag = sentence[i]
            try:
                A_prob = pow(10, A[(START, tag)])
            except KeyError:
                A_prob = 1/(num_of_sentences + V)
            try:
                B_prob = pow(10, B[(tag, word)])
            except KeyError:
                tag_counts = 0 if tag not in allTagCounts else allTagCounts[tag]
                B_prob = 1/(tag_counts + V)
            p *= A_prob * B_prob
        else:
            pre_word, pre_tag = sentence[i - 1]
            curr_word, curr_tag = sentence[i]
            try:
                A_prob = pow(10, A[(pre_tag, curr_tag)])
            except KeyError:
                pre_tag_counts = 0 if pre_tag not in allTagCounts else allTagCounts[pre_tag]
                A_prob = 1/(pre_tag_counts + V)
            try:
                B_prob = pow(10, B[(curr_tag, curr_word)])
            except KeyError:
                curr_tag_counts = 0 if curr_tag not in allTagCounts else allTagCounts[curr_tag]
                B_prob = 1/(curr_tag_counts + V)
            p *= A_prob * B_prob
    try:
        last_word, last_tag = sentence[-1]
        p *= pow(10, A[(last_tag, END)])
    except KeyError:
        p *= 1 / (num_of_sentences + V)
    p = log(p, 10)

    assert isfinite(p) and p<0  # Should be negative. Think why!
    return p


#===========================================================
#       Wrapper function (tagging with a specified model)
#===========================================================

def tag_sentence(sentence, model):
    """Returns a list of pairs (w,t) where pair corresponds to a word (same index) in
    the input sentence. Tagging is done with the specified model.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict): a dictionary where key is the model name and the value is
           an ordered list of the parameters of the trained model (baseline, HMM).

        Models that must be supported (you can add more):
        1. baseline: {'baseline': [perWordTagCounts, allTagCounts]}
        2. HMM: {'hmm': [A,B]}


        The parameters for the baseline model are:
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        The parameters for the HMM are:
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.


    Return:
        list: list of pairs
    """
    if list(model.keys())[0]=='baseline':
        return baseline_tag_sentence(sentence, list(model.values())[0][0], list(model.values())[0][1])
    if list(model.keys())[0]=='hmm':
        return hmm_tag_sentence(sentence, list(model.values())[0][0], list(model.values())[0][1])


def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags,the total number of
    correcttly predicted tags for oov words and the number of oov words in the
    given sentence.

    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger

    """
    assert len(gold_sentence)==len(pred_sentence)

    #TODO complete the code
    global START,END, UNK, allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B, num_of_sentences
    correct, correctOOV = 0, 0
    for (gold_word, gold_tag), (pred_word, pred_tag) in zip(gold_sentence, pred_sentence):
        if gold_tag == pred_tag:
            correct += 1
            if pred_word not in perWordTagCounts:
                correctOOV += 1
    OOV = len([word for word, tag in pred_sentence if word not in perWordTagCounts])
    return correct, correctOOV, OOV


if __name__ == '__main__':
    main()