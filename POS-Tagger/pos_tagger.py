class Color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

import nltk
import random, operator
from collections import defaultdict
from nltk.tokenize import sent_tokenize,word_tokenize

# this function generates POS tags using NLTK's utility, for training HMM
# instead of all possible tags, the following 12 tags (universal) are considered
#   Noun (noun), Verb (verb), Adj (adjective), Adv (adverb), Pron (pronoun), 
#   Det (determiner or article), Adp (preposition or postposition), Num (numeral), 
#   Conj (conjunction), Prt (particle), ‘.’ (punctuation mark) and x (other).  
# INPUT: path of input corpus (if nothing mentioned, Brown corpus is taken)
# OUTPUT: list of lists of POS tags, with each inner list correponding to a sentence
def load_corpus(path=None):
    if not path:    # load from Brown corpus
        # this is a list of list of tagged words (inner list is for a sentence)
        tagged_sentences=list(nltk.corpus.brown.tagged_sents(tagset='universal'))
        print("\nTotal number of tagged sentences: ",len(tagged_sentences))
        return tagged_sentences
    
    # reach here if the path is defined
    with open(path,"r") as input_corpus:
        # read the entire input file as a string replacing \n with blank space
        file_str=input_corpus.read().replace('\n', ' ')

        # since we require a list of lists of tags (corresponding to sentences),
        #   tokenize by sentences first and later word_tokenize
        sentences=sent_tokenize(file_str)
        
        tagged_sentences=[]
        for sentence in sentences:
            words=word_tokenize(sentence)
            tagged_sentences.append(nltk.pos_tag(words,tagset='universal'))
        
        print("\nTotal number of tagged sentences: ",len(tagged_sentences))
        return tagged_sentences

# this is to initialize all the parameters required by the HMM
#   transition probs, emission probs, intial state probs
# INPUT: tagged sentences from above function
# OUTPUT: all required params for HMM
def init_HMM(tagged_sentences):
    # FINDING INITIAL STATE PROBS
    # the max size of this dictionary can be 12 (corresponding to each POS tag)
    initial_state_probs=defaultdict(lambda: 0)
    for tagged_sentence in tagged_sentences:
        # pick the starting tag for each sentence and maintain its count
        initial_state_probs[tagged_sentence[0][1]]+=1
    # the total number of times any tag appears in the beginning
    denominator=sum(initial_state_probs.values())
    # calculate probability for each tag
    for k,v in initial_state_probs.items():
        initial_state_probs[k]/=denominator

    # FINDING TRANSITION PROBS
    # this dictionary can have a max of 12 rows and 13 cols,
    #   the last col being for <E>, i.e. end of sentence
    transition_probs=defaultdict(lambda: defaultdict(lambda: 0))
    for tagged_sentence in tagged_sentences:
        for i in range(len(tagged_sentence)-1):
            transition_probs[tagged_sentence[i][1]][tagged_sentence[i+1][1]]+=1
        # for last word, consider <E> as the end of sentence marker
        last_word_index=len(tagged_sentence)-1
        transition_probs[tagged_sentence[last_word_index][1]]['<E>']+=1
    # calculate probability for each transition
    for k,inner_dict in transition_probs.items():
        denominator=sum(inner_dict.values())
        for k1,v in inner_dict.items():
            transition_probs[k][k1]/=denominator

    # FINDING EMISSION PROBS
    # this dictionary can have a max of 12 rows (cols correspond to words)
    emission_probs=defaultdict(lambda: defaultdict(lambda: 0))
    for tagged_sentence in tagged_sentences:
        for i in range(len(tagged_sentence)):
            emission_probs[tagged_sentence[i][1]][tagged_sentence[i][0].lower()]+=1
    # calculate probability for each emission
    for k,inner_dict in emission_probs.items():
        denominator=sum(inner_dict.values())
        for k1,v in inner_dict.items():
            emission_probs[k][k1]/=denominator

    return initial_state_probs,transition_probs,emission_probs

# function to find the most probable tag for each word in the input
# 'words' is a sentence in the form of a list of words
def most_probable_tag(words,*hmm_params):
    initial_state_probs=hmm_params[0]
    transition_probs=hmm_params[1]
    emission_probs=hmm_params[2]

    # list of all 12 tags
    tags=['ADJ','ADP','ADV','CONJ','DET','NOUN','NUM','PRT','PRON','VERB','.','X']

    # the viterbi algorithm is used here to find the most probable tag
    # a dp table for probabilities (as required in forward algo) is maintained
    # consider a word sequence V1,V3,V2
    #   at t = 1 (V1 to be generated),
    #       the probability for state Qi is calculated as (initial_state_probs)[i]*(emission_probs)[i,V3]
    #   for t from 2 upto 3 (to generate V3 and then V2),
    #       probability for all states is calulated bottom-up (from previous t)
    
    # since we require just the path and not the final probability,
    # there is no need to incorporate final state (<E>)
    
    alpha=defaultdict(lambda: defaultdict(lambda: 0))   # the DP table
    num_words=len(words)
    result_tags=[]  # result tags

    # t = 1
    for tag in tags:
        alpha[1][tag]=initial_state_probs[tag]*emission_probs[tag][words[0].lower()]
    # this is to find the tag with max probability at t=1
    max_prob=max(alpha[1].items(), key=operator.itemgetter(1))
    if max_prob[1]==0:  # if probability value is zero, tag cannot be allocated
        result_tags.append("Tag not found!")
    else:
        result_tags.append(max_prob[0])

    # t = 2 to num_words
    for t in range(2,num_words+1):
        word=words[t-1].lower() # the word to be generated at this time
        
        for tag in tags:    # fill entry for 'tag' at time 't'
            emission_prob_here=emission_probs[tag][word]
            # find the sum of alpha(i)(t-1)*(a)ij for all i
            sum=0
            for tag_prev in tags:
                sum+=alpha[t-1][tag_prev]*transition_probs[tag_prev][tag]
            alpha[t][tag]=sum*emission_prob_here

        # this is to find the tag with max probability at time 't'
        max_prob=max(alpha[t].items(), key=operator.itemgetter(1))
        if max_prob[1]==0:  # if probability value is zero, tag cannot be allocated
            result_tags.append("Tag not found!")
        else:
            result_tags.append(max_prob[0])
    
    return list(zip(words,result_tags))


"""
DRIVER CODE
"""
input_file=input("Do you want to train using an input file (enter file path) or Brown corpus (simply press enter) ?\n")
# load the training corpus and initialize HMM params
print("\n========== LOADING CORPUS ==========")
tagged_sentences=load_corpus(input_file)
print("\n========== INITIALIZING HMM ==========")
initial_state_probs,transition_probs,emission_probs=init_HMM(tagged_sentences)

with open("test_sentences.txt","r") as test_file:
    test_sentences=sent_tokenize(test_file.read().replace("\n"," "))
    # take each sentence from the test file and tag it
    for test_sentence in test_sentences:
        print()
        print(test_sentence)
        words=[word.lower() for word in word_tokenize(test_sentence)]
        result_tags=most_probable_tag(words,initial_state_probs,transition_probs,emission_probs)
        print(Color.BOLD,result_tags,Color.END)
