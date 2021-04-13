# mp4.py


"""
This file should not be submitted - it is only meant to test your implementation of the Viterbi algorithm. 

See Piazza post - This example is intended to show you that even though P("back" | RB) > P("back" | VB), 
the Viterbi algorithm correctly assigns the tag as VB in this context based on the entire sequence. 
"""
from utils import read_files, get_nested_dictionaries
import math

def main():
    test, emission, transition, output = read_files()
    emission, transition = get_nested_dictionaries(emission, transition)
    initial = transition["START"]
    prediction = []
    
    """WRITE YOUR VITERBI IMPLEMENTATION HERE"""

    print('Your Output is:',prediction,'\n Expected Output is:',output)


if __name__=="__main__":
    main()