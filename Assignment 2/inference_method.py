from collections import deque
from support import definite_clause

### THIS IS THE TEMPLATE FILE
### WARNING: DO NOT CHANGE THE NAME OF FILE OR THE FUNCTION SIGNATURE

def pl_fc_entails(symbols_list : list, KB_clauses : list, known_symbols : list, query : int) -> bool:
    """
    pl_fc_entails function executes the Propositional Logic forward chaining algorithm (AIMA pg 258).
    It verifies whether the Knowledge Base (KB) entails the query
        Inputs
        ---------
            symbols_list  - a list of symbol(s) (have to be integers) used for this inference problem
            KB_clauses    - a list of definite_clause(s) composed using the numbers present in symbols_list
            known_symbols - a list of symbol(s) from the symbols_list that are known to be true in the KB (facts)
            query         - a single symbol that needs to be inferred

            Note: Definitely check out the test below. It will clarify a lot of your questions.

        Outputs
        ---------
        return - boolean value indicating whether KB entails the query
    """

    ### START: Your code

    count = {} # dictionary to hold the number of symbols in each clause
    for c in KB_clauses:
        count[c] = len(c.body)
    inferred = [False for i in range(max(symbols_list)+1)] # everything initially assumed false
    agenda = known_symbols # agenda queue to follow psuedo code

    while(agenda): # loop while the agenda queue is not empty
        p = agenda.pop()
        if p == query:
            return True # if the symbol is in the query inputted, then it must be true
        elif inferred[p] == False:
            inferred[p] = True # change inferred from false to true since we now know its state
            for c in KB_clauses: # loop through the various clauses
                if p in c.body:
                    count[c] -= 1 # update the number of symbols as the loop continues
                    if count[c] == 0:
                        agenda.append(c.conclusion) # all symbols are true --> conclusion is now known

    return False # remove line if needed
    ### END: Your code


# SAMPLE TEST
if __name__ == '__main__':

    # Symbols used in this inference problem (Has to be Integers)
    symbols = [1,2,9,4,5]

    # Clause a: 1 and 2 => 9
    # Clause b: 9 and 4 => 5
    # Clause c: 1 => 4
    KB = [definite_clause([1, 2], 9), definite_clause([9,4], 5), definite_clause([1], 4)]

    # Known Symbols 1, 2
    known_symbols = [1, 2]

    # Does KB entail 5?
    entails = pl_fc_entails(symbols, KB, known_symbols, 5)

    print("Sample Test: " + ("Passed" if entails == True else "Failed"))
