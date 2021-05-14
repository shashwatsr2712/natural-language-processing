# utility class for pretty-printing, if needed
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
from nltk.tokenize import word_tokenize
from nltk.corpus import treebank
from nltk import Nonterminal,induce_pcfg
from nltk import parse,grammar
from nltk.featstruct import Feature,FeatStruct
from collections import defaultdict


# GRAMMAR RULES FOR DIFFERENT STRUCTURES OF ENGLISH SENTENCES

# Declarative:
# S->NP VP

# Imperative:
# S->VP

# Yes-No Questions:
# S->Aux NP VP

# Wh Questions:
# S->Wh NP VP | Wh NP Aux NP VP

# NP->Det Nominal | ProperNoun
# Nominal->Noun | Noun Nominal
# VP->Verb | Verb NP | Verb NP PP | Verb PP
# PP->Preposition NP
# Det->'a' | 'the'
# Noun->...
# Verb->...
# ProperNoun->...
# Preposition->...


# NOTE:
# to get all rules for a grammar, use grammar.productions()
# for each rule,
#   grammar.productions()[i].lhs() {or rhs()}, grammar.productions()[i].prob()
# type of non-terminals is 'Nonterminal', that of terminal is 'str'  
# lhs is of type 'Nonterminal', rhs is a tuple of terminals and/or non-terminals 

# a custom class for each cell in the parsing DP table
# each cell in the DP table will be a dictionary with key as a Nonterminal
#   and value as an object of this class 
class CKYCellData():
    # k_value tells from where the maximum parse for a non terminal is taken
    # for instance, k_value=3 for cell 2,4 tells that it is taken from 2,3 and 4,4
    #   in general, for cell (i,j), (i,k_value) and (k_value+1,j) are the backpointers
    # non_term1 and non_term2 are the non terminals for the backpointers to (i,k_value) and (k_value+1,j) respectively
    # i.e. for A->BC in cell (i,j), non_term1 will be B, and non_term2 C
    def __init__(self,prob,k_value,non_term1,non_term2):
        self.prob=prob
        self.k_value=k_value
        self.non_term1=non_term1
        self.non_term2=non_term2
        
# function to compute the CKY parsing table for a given sentence and CNF grammar
# INPUT: sentence to parse as a list of tokens, grammar in CNF
# OUTPUT: filled CKY parsing table (upper triangular matrix)
def CKY_parser(sentence,grammar):
    num_words=len(sentence)

    # the DP table for parsing (rows and columns are 0-indexed)
    # this is a list of lists (essentially, a matrix with cells)
    #   with each cell containing a dictionary
    #   where the key value is a Nonterminal, and the value is an object containing
    #   probability value, the k_value, and the 2 non_terminals for backpointers 
    parsing_dp=[[dict() for i1 in range(num_words)] for i2 in range(num_words)]
    
    # initialization of principal diagonal entries
    for j in range(num_words):
        # iterate over each of the production rules
        for rule in grammar.productions():
            # if the rule is of the form A->a and 'a' is the word at index 'j',
            #   include 'A' in this cell of the table
            if len(rule.rhs())==1 and rule.rhs()[0].lower()==sentence[j].lower():
                # the k_value and non-terminals in CKYCellData are meaningless here
                parsing_dp[j][j][rule.lhs()]=CKYCellData(rule.prob(),j,Nonterminal('<<>>'),Nonterminal('<<>>'))

    # filling the rest of the table
    # start with columns from left to right, and for each column, go bottom-up
    for j in range(1,num_words):
        for i in range(j-1,-1,-1):
            # check if Nonterminals at (i,k) and (k+1,j) can form RHS of a rule
            for k in range(i,j):
                # iterate over each grammar rule
                for rule in grammar.productions():
                    # the rule should be of of the form A->BC and 
                    # B & C should be at (i,k+1) & (k+1,j) respectively
                    if len(rule.rhs())==2 and (rule.rhs()[0] in parsing_dp[i][k]) and (rule.rhs()[1] in parsing_dp[k+1][j]):
                        # get the probabilities of parsing_dp[i][k] and parsing_dp[k+1][j]
                        prob_product=parsing_dp[i][k][rule.rhs()[0]].prob*parsing_dp[k+1][j][rule.rhs()[1]].prob
                        # if 'A' is already in this cell with more probability, just continue
                        if rule.lhs() in parsing_dp[i][j] and parsing_dp[i][j][rule.lhs()].prob>=rule.prob()*prob_product:
                            continue
                        # reach here if 'A' is either absent or present with lower probability
                        parsing_dp[i][j][rule.lhs()]=CKYCellData(rule.prob()*prob_product,k,rule.rhs()[0],rule.rhs()[1])
    
    return parsing_dp

# utility to print the parse tree of sentence from CKY table
def printTree(p_file,table,sentence):
    num_words=len(table)
    p_file.write("Probability of the most probable parse tree for the sentence:\n"+sentence+"\nis "+str(table[0][num_words-1][Nonterminal('S')].prob)+"\n\n")
    printTreeUtil(p_file,word_tokenize(sentence),table,0,num_words-1,Nonterminal('S'),0)

# utility called by 'printTree(A,B)'
# INPUT: output file,list of words in sentence, CKY table, current row, current column,
#   current non-terminal, indentation for printing tabs before printing the non-terminal
def printTreeUtil(f,words,table,r,c,non_terminal,indent):
    for i in range(indent):
        f.write("\t")
    
    if r==c:    # base case => print the terminal symbol (word of sentence)
        f.write(str(non_terminal)+"\t")
        f.write(words[r]+"\n")
        return
    
    f.write(str(non_terminal)+"\n")
    # get the k_value for current location's non-terminal
    # also, from the same dictionary, get the non-terminals for both the backpointers
    k_val=table[r][c][non_terminal].k_value
    left_child_non_terminal=table[r][c][non_terminal].non_term1
    right_child_non_terminal=table[r][c][non_terminal].non_term2
    
    # recurse on both children
    printTreeUtil(f,words,table,r,k_val,left_child_non_terminal,indent+1)
    printTreeUtil(f,words,table,k_val+1,c,right_child_non_terminal,indent+1)

# NOTE:
# to extract feature structure and symbols from lhs and rhs of Feature Grammar:- 
# for p in feature_grammar_productions:
#     lhs_nonterminal=Nonterminal(p.lhs()[Feature('type')])
#     if 'AGR' in p.lhs():
#         lhs_featstruct=FeatStruct(AGR=p.lhs()['AGR'])
#     if len(p.rhs())==1: # for rules of type A->a
#         rhs_string=p.rhs()[0]
#     else:
#         rhs_nonterminal1=Nonterminal(p.rhs()[0][Feature('type')])
#         rhs_nonterminal2=Nonterminal(p.rhs()[1][Feature('type')])
#         if 'AGR' in p.rhs()[0]:
#             rhs_featstruct1=FeatStruct(AGR=p.rhs()[0]['AGR'])
#         if 'AGR' in p.rhs()[1]:
#             rhs_featstruct2=FeatStruct(AGR=p.rhs()[1]['AGR'])

# function to parse using CKY algorithm for a given sentence and CNF Feature Grammar
# INPUT: sentence to parse as a list of tokens, feature grammar in CNF
# OUTPUT: filled CKY parsing table (upper triangular matrix) or,
#   -1 with error message in case of disagreement
def feature_CKY_parser(sentence,grammar):
    num_words=len(sentence)

    # the DP table for parsing (rows and columns are 0-indexed)
    # this is a list of lists (essentially, a matrix with cells)
    #   with each cell containing a dictionary
    #   where the key value is a Nonterminal, and the value is a set containing
    #   AGREEMENT feature structures for this non-terminal 
    parsing_dp=[[defaultdict(lambda:set()) for i1 in range(num_words)] for i2 in range(num_words)]
    
    # initialization of principal diagonal entries
    for j in range(num_words):
        # iterate over each of the production rules
        for rule in grammar.productions():
            # if the rule is of the form A->a and 'a' is the word at index 'j',
            #   include 'A' in this cell of the table
            if len(rule.rhs())==1 and rule.rhs()[0].lower()==sentence[j].lower():
                lhs_nonterminal=Nonterminal(rule.lhs()[Feature('type')])
                lhs_featstruct=FeatStruct()
                if 'AGR' in rule.lhs():
                    lhs_featstruct=FeatStruct(AGR=rule.lhs()['AGR'])
                FeatStruct.freeze(lhs_featstruct)   # freeze to make it hashable
                parsing_dp[j][j][lhs_nonterminal].add(lhs_featstruct)
        
    # filling the rest of the table
    # start with columns from left to right, and for each column, go bottom-up
    for j in range(1,num_words):
        for i in range(j-1,-1,-1):
            # check if Nonterminals at (i,k) and (k+1,j) can form RHS of a rule
            for k in range(i,j):
                # iterate over each grammar rule
                for rule in grammar.productions():
                    # the rule should be of of the form A->BC and 
                    # B & C should be at (i,k+1) & (k+1,j) respectively
                    if len(rule.rhs())==2 and (Nonterminal(rule.rhs()[0][Feature('type')]) in parsing_dp[i][k]) and (Nonterminal(rule.rhs()[1][Feature('type')]) in parsing_dp[k+1][j]):
                        possible=False
                        lhs_nonterminal=Nonterminal(rule.lhs()[Feature('type')])
                        rhs_nonterminal1=Nonterminal(rule.rhs()[0][Feature('type')])
                        rhs_nonterminal2=Nonterminal(rule.rhs()[1][Feature('type')])
                        
                        if 'AGR' in rule.rhs()[0] and 'AGR' in rule.rhs()[1]:
                            # for all the different feature structures for B and C,
                            # unification should be possible for at least one 
                            for f1 in parsing_dp[i][k][rhs_nonterminal1]:
                                for f2 in parsing_dp[k+1][j][rhs_nonterminal2]:
                                    f3=f1.unify(f2)
                                    if f3 is not None:
                                        possible=True
                                        FeatStruct.freeze(f3)   # freeze to make it hashable
                                        # copy the unified agreement to A 
                                        parsing_dp[i][j][lhs_nonterminal].add(f3)
                        elif 'AGR' in rule.rhs()[0] and 'AGR' in rule.lhs():
                            # just copy the AGREEMENTs from B tp A
                            possible=True
                            for f in parsing_dp[i][k][rhs_nonterminal1]:
                                parsing_dp[i][j][lhs_nonterminal].add(f)
                        else:
                            possible=True
                            pass

                        # if the value of 'possible' remains False, it implies that
                        #   no unification of FeatStructs is possible for B,C in A->BC 
                        if not possible:
                            print("Disgreement in features for the rule:\n",rule)
                            return -1
    
    return parsing_dp


"""
DRIVER CODE
"""

# PCFG PARSING USING CKY
# production rules from treebank corpus
productions=[]
# consider first 100 files for training (some issue in file 55)
training_fileids=treebank.fileids()[:55]+treebank.fileids()[56:100]
for item in training_fileids:   # iterate over all training file ids
    # 'item' will be a file containing one or more sentences which we need to consider
    for tree in treebank.parsed_sents(item):    # iterate over each sentence's parse tree
        # collapse unary productions (like A->B) and convert the tree to CNF
        # before getting all the production rules
        tree.collapse_unary(collapsePOS=True)
        tree.chomsky_normal_form()
        p=tree.productions()
        productions+=p

# get a PCFG from the above formed production rules, with start Nonterminal as 'S'
pcfg_grammar=induce_pcfg(Nonterminal('S'),productions)

# test sentences (each as a string and then tokenized)
# taken from file or from treebank (comment either of the two below)
test_sentences=[]
# # take from treebank
# for f in treebank.fileids()[100:105]:
#     test_sentences.append(" ".join(treebank.parsed_sents(f)[0].leaves()))
# take from file
with open("test_sentences.txt","r") as test_file:
    for line in test_file:
        test_sentences.append(line)
# get the parse result for each test sentence and print the parse tree to a file
parse_file=open("parse_file.txt","w")
for test_sentence in test_sentences:
    test_sentence_tokens=word_tokenize(test_sentence)
    res=CKY_parser(test_sentence_tokens,pcfg_grammar)   # get the result of CKY parser
    # print the parse tree to file
    printTree(parse_file,res,test_sentence)
    parse_file.write("\n")
parse_file.close()


# INCORPORATING SEMANTIC RULES FOR AGREEMENT USING FEATURE STRUCTURES
# nltk.download('book_grammars')
# cp=parse.load_parser('grammars/book_grammars/feat0.fcfg',trace=1)
# nltk.data.show_cfg('grammars/book_grammars/feat0.fcfg')

# refer to the file 'feature_grammar.txt' for details
g1="""
S -> N[AGR=[NUM=?n, PER=?p]] VP[AGR=[NUM=?n, PER=?p]]
S -> PropN[AGR=[NUM=?n, PER=?p]] VP[AGR=[NUM=?n, PER=?p]]
S -> NP[AGR=[NUM=?n, PER=?p]] VP[AGR=[NUM=?n, PER=?p]]
NP[AGR=[NUM=?n, PER=?p]] -> Det[AGR=[NUM=?n]] N[AGR=[NUM=?n, PER=?p]]
VP[AGR=[TENSE=?t, NUM=?n, PER=?p]] -> IV[AGR=[TENSE=?t, NUM=?n, PER=?p]] Dot
VP[AGR=[TENSE=?t, NUM=?n, PER=?p]] -> TVPar[AGR=[TENSE=?t, NUM=?n, PER=?p]] Dot
TVPar[AGR=[TENSE=?t, NUM=?n, PER=?p]] -> TV[AGR=[TENSE=?t, NUM=?n, PER=?p]] NP
TVPar[AGR=[TENSE=?t, NUM=?n, PER=?p]] -> TV[AGR=[TENSE=?t, NUM=?n, PER=?p]] N
Dot -> '.'
Det[AGR=[NUM=sg]] -> 'this' | 'every'
Det[AGR=[NUM=pl]] -> 'these' | 'all'
Det -> 'the' | 'some' | 'several'
PropN[AGR=[NUM=sg, PER=3]]-> 'Josh' | 'Swain'
N[AGR=[NUM=sg, PER=3]] -> 'dog' | 'girl' | 'car' | 'child'
N[AGR=[NUM=pl, PER=3]] -> 'dogs' | 'girls' | 'cars' | 'children'
N[AGR=[NUM=sg, PER=1]] -> 'I'
N[AGR=[NUM=pl, PER=1]] -> 'we'
N[AGR=[PER=2]] -> 'you'
IV[AGR=[TENSE=pres,  NUM=sg, PER=3]] -> 'disappears' | 'walks'
TV[AGR=[TENSE=pres, NUM=sg, PER=3]] -> 'sees' | 'likes'
IV[AGR=[TENSE=pres,  NUM=pl, PER=3]] -> 'disappear' | 'walk'
TV[AGR=[TENSE=pres, NUM=pl, PER=3]] -> 'see' | 'like'
IV[AGR=[TENSE=pres, PER=1]] -> 'disappear' | 'walk'
TV[AGR=[TENSE=pres, PER=1]] -> 'see' | 'like'
IV[AGR=[TENSE=pres, PER=2]] -> 'disappear' | 'walk'
TV[AGR=[TENSE=pres, PER=2]] -> 'see' | 'like'
IV[AGR=[TENSE=past]] -> 'disappeared' | 'walked'
TV[AGR=[TENSE=past]] -> 'saw' | 'liked'
"""
feature_grammar=grammar.FeatureGrammar.fromstring(g1)

sent='Josh likes these children .'  # sentence to parse
tokens=sent.split()

print("===== Parsing using NLTK's EarleyChartParser (Parse Trees will be printed for successful parsing) =====\n")
parser=nltk.FeatureEarleyChartParser(feature_grammar)
parsed=False    # boolean to determine if successful parsing (agreement) is done
trees=parser.parse(tokens)
# iterate over all parse trees and draw/print them
for tree in trees:
    parsed=True
    tree.draw()
    print(tree)
if not parsed:
    print("Disagreement in features detected!")

print("\n===== Parsing using the implemented CKY parser =====\n")
if feature_CKY_parser(tokens,feature_grammar)!=-1:
    print("Succesfully parsed!")


# # DISCARDED
# from collections import defaultdict
# # dictionary to store (eventually) the prob. associated with each rule
# # initially it contains the count of each rule in the corpus
# tbank_productions_prob=defaultdict(lambda:0)
# for production in tbank_productions:
#     tbank_productions_prob[production]+=1
# # calculate prob. of each rule from their counts in the corpus
# for production,count in tbank_productions_prob.items():
#     total_count_for_lhs=sum(val for key,val in tbank_productions_prob.items() if key.lhs()==production.lhs())
#     tbank_productions_prob[production]=count/total_count_for_lhs

# tbank_productions=list(production for sent in treebank.parsed_sents() for production in sent.productions())

# for p in grammar.productions():
#     if len(p.rhs())==1 and isinstance(p.rhs()[0],nltk.grammar.Nonterminal):
#         print("A")
