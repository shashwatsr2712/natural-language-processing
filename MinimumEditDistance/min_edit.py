# Custom class for data during computation of results
class ValDir():
    def __init__(self,val,is_inserted,is_deleted,is_substituted):
        self.val=val
        self.is_inserted=is_inserted
        self.is_deleted=is_deleted
        self.is_substituted=is_substituted


# Calculate minimum edit distance and return the whole DP table
def get_min_edit_distance_table(s1,s2):
    len1,len2=len(s1),len(s2)

    # DP table to compute and store result in bottom-up manner
    num_rows,num_cols=len1+1,len2+1
    track_edits=[[ValDir(0,False,False,False) for i in range(num_cols)] for i in range(num_rows)]

    # Initialization of DP table
    for i in range(1,num_cols):   # trying to construct s2[1...i] from nothing in s1
        track_edits[0][i].val=track_edits[0][i-1].val+1;    # For insertion
        track_edits[0][i].is_inserted=True

    for i in range(1,num_rows):   # trying to construct empty string s2 from s1[1...i]
        track_edits[i][0].val=track_edits[i-1][0].val+1 # For deletion
        track_edits[i][0].is_deleted=True

    # Computation of DP table
    for i in range(1,len1+1):
        for j in range(1,len2+1):
            swap_cost=2
            if s1[i-1]==s2[j-1]:
                swap_cost=0

            """
                trying to convert s1[1...i] to s2[1...j] from the solution of one of the follows:
                -> s1[1...(i-1)] to s2[1...j] by deleting ith character of s1
                -> s1[1...i] to s2[1...(j-1)] by inserting jth character of s2
                -> s1[1...(i-1)] to s2[1...(j-1)] by substitution of ith char of s1 and jth of s2

                Hence,
                -> substitution moves both pointers ahead
                -> insertion moves the s2 pointer ahead
                (moving horizontally to lower column in table)
                -> deletion moves the s1 pointer ahead
                (moving vertically to lower row in table)
            """

            min_val=min(
                track_edits[i-1][j].val+1,
                min(
                    track_edits[i][j-1].val+1,
                    track_edits[i-1][j-1].val+swap_cost
                )
            )
            track_edits[i][j].val=min_val

            if min_val-1==track_edits[i-1][j].val:
                track_edits[i][j].is_deleted=True
            if min_val-1==track_edits[i][j-1].val:
                track_edits[i][j].is_inserted=True
            if min_val-swap_cost==track_edits[i-1][j-1].val:
                track_edits[i][j].is_substituted=True

    return track_edits

# Utility to find all the different optimal alignments
# in terms of insertion (I), deletion (D), and substitution (S)
# using the already computed DP table
def computeWays(res,temp,r,c,track_edits):
    if r==0 and c==0:
        res.append(temp)
        return

    if track_edits[r][c].is_inserted:
        computeWays(res,"I"+temp,r,c-1,track_edits)

    if track_edits[r][c].is_deleted:
        computeWays(res,"D"+temp,r-1,c,track_edits)

    if track_edits[r][c].is_substituted:
        computeWays(res,"S"+temp,r-1,c-1,track_edits)

# Utility to find all the different optimal alignments
# in terms of actual position of the characters in the two strings
# using the already computed DP table
def computeWaysPrint(temp1,temp2,r,c,track_edits,s1,s2):
    if r==0 and c==0:
        print(temp1)
        print(temp2)
        print("===================\n")
        return

    if track_edits[r][c].is_inserted:
        computeWaysPrint("_"+temp1,s2[c-1]+temp2,r,c-1,track_edits,s1,s2)

    if track_edits[r][c].is_deleted:
        computeWaysPrint(s1[r-1]+temp1,"_"+temp2,r-1,c,track_edits,s1,s2)

    if track_edits[r][c].is_substituted:
        computeWaysPrint(s1[r-1]+temp1,s2[c-1]+temp2,r-1,c-1,track_edits,s1,s2)

"""
DRIVER CODE
"""
s1,s2=input("Enter the two words (space-separated) for minimum edit distance: ").split(" ")
len1,len2=len(s1),len(s2)
track_edits=get_min_edit_distance_table(s1,s2)
print("Word 1 (s1): ",s1)
print("Word 2 (s2): ",s2)
print("\nMinimum Edit distance: ",track_edits[len1][len2].val)

choice=int(input("\nDo you want to see all possible alignments (Enter 1) or the operations to be performed in terms of S,I,D (Enter 2 [default]) ? "))
if choice==1:
    print()
    computeWaysPrint("","",len1,len2,track_edits,s1,s2)
else:
    res=list()
    # Compute all the possible ways (in terms of S,I,D)
    # by backtrace from the target index in DP table
    computeWays(res,"",len1,len2,track_edits)

    # Of all the found ways, let us consider those with minimum "number" of operations 
    min_operations=999999
    for temp in res:
        min_operations=min(min_operations,len(temp))

    # Print all the ways with minimum distance AND minimum operations
    print("\nWays to convert s1 to s2 with minimum number of operations:")
    for temp in res:
        if len(temp)==min_operations:
            print(temp)

print("\nPlease wait while autocorrect is loading...")

# SPELL-CHECK USING A CATEGORY OF BROWN CORPUS
from nltk.corpus import brown
dictionary=set()
for word in brown.words(categories='news'):
    word=word.lower()
    dictionary.add(word)

print("\n=== AUTOCORRECT ===")
while True:
    entered_word=input("\nEnter a word (-1 to quit): ").strip().lower()
    len_entered=len(entered_word)
    if entered_word=="-1":
        break
    if entered_word in dictionary:
        print("This word is valid!")
        continue

    # Find the minimum of minimum-edit-distance from all words in dictionary
    min_dist=99999
    for word in dictionary:
        len_check=len(word)
        temp_table=get_min_edit_distance_table(entered_word,word)
        min_dist=min(min_dist,temp_table[len_entered][len_check].val)
    
    # Upper bound for minimum-edit-distance
    if min_dist>4:
        print("No close match found!")
        continue
    
    # For all the words with minimum minimum-edit-distance
    # find the least absolute difference between the length of it and input
    min_length_diff=99999
    for word in dictionary:
        len_check=len(word)
        temp_table=get_min_edit_distance_table(entered_word,word)
        if temp_table[len_entered][len_check].val==min_dist:
            length_diff=abs(len_check-len_entered)
            min_length_diff=min(min_length_diff,length_diff)
    
    # Finally, collect all the words with both minimum minimum-edit-distance
    # and least absolute difference in size
    closest_words=[]
    for word in dictionary:
        len_check=len(word)
        temp_table=get_min_edit_distance_table(entered_word,word)
        if temp_table[len_entered][len_check].val==min_dist and abs(len_check-len_entered)==min_length_diff:
            closest_words.append(word)
    
    print("Did you mean the following?")
    for word in closest_words:
        print(word)