# Joseph Chang - G01189913

# Data Visualization
from re import A
from threading import active_count
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import time

# File paths
training_file = "1645760197_9666755_1568904114_9130223_train_drugs_1.txt"
testing_file = "1645760197_9795408_1568904114_9234657_test.txt"
output_file = "hw2_output.txt"


# *** Naive Bayes Implementation ***

# counts of training records classified as either active or inactive
active_count = 0
inactive_count = 0

# set of active and inactive records' inputs
active_set = {}
inactive_set = {}

"""
Naive Bayes Training Segment
This train_nb() function is the training sequence for the naive bayes approach
The higher probability of classification (or p(C)) given the probability of the given record inputs (or p(C|x)) is equivalent to p(C)*p(x|C)/p(x) where C is the classification (as 0 or 1) and x are the record inputs (indices)
"""
def train_nb():
    # global fields
    global active_count
    global inactive_count
    global active_set
    global inactive_set

    # open training file
    train_file = open(training_file, "r", encoding="utf-8")
    
    # read each line of the training file
    while (1):
        line = train_file.readline()
        if (line == ""): # ""/eof -> done reading lines
            break
        
        # get the input data of the current line's record
        input = line[2:].split(" ")[:-1] # the last index is an empty string
        # get the active status of current line's record
        active = int(line[0]) # 0 for inactive : 1 for active
        for x in input:
            if active == 0:
                inactive_count += 1
                if x in inactive_set:
                    inactive_set[x] += 1
                else:
                    inactive_set[x] = 1
            elif active == 1:
                active_count += 1
                if x in active_set:
                    active_set[x] += 1
                else:
                    active_set[x] = 1
            else:
                print("error parsing active (not 0 or 1)")
        
        # END TRAINING FILE READ

    # close train files
    train_file.close()
    
"""
Naive Bayes Testing Segment
Naive Bayes Assumption: each feature is independent and equal (or at least close to it)
Implementation notes:
- assumes test class distribution follows the training class distribution (important for calculation the "prior" values)
- feature reduction on features that seem to be too redundant or too generic between all records
"""
def test_nb():
    # open testing file
    test_file = open(testing_file, "r", encoding="utf-8")
    out_file = open(output_file, "w", encoding="utf-8")
    output_string = ""
    
    # prior value calculations
    # prior value calculated assuming testing set has same distribution as training set
    prior0 = inactive_count / (active_count + inactive_count)
    prior1 = active_count / (active_count + inactive_count)
    #prior0 = prior1 = 1
    print(f"prior0 = {prior0}\nprior1 = {prior1}")

    # analysis fields
    tie_count = 0
    set_null_count = 0

    # read each line of the testing file
    while (tie_count == 0):
        line = test_file.readline()
        if (line == ""): # ""/eof -> done reading lines
            break
        
        # get the input data of the current line's record
        input = line.split(" ")[:-1] # the last index is an empty string
        
        # prior0 and prior1 should be defined after training
        # Calculating likelihood (or conditional probability) of the current given record
        likelihood0 = 1.0 # p(x|C_0)
        likelihood1 = 1.0 # p(x|C_1)
        for x in input:
            if x in inactive_set:
                likelihood0 *= (inactive_set[x] / len(inactive_set))
            else:
                set_null_count += 1
                likelihood0 *= 1.0
            if x in active_set:
                likelihood1 *= (active_set[x] / len(active_set))
            else:
                set_null_count += 1
                likelihood1 *= 1.0


        # evidence can be ignored (arbitrary) because it is the same for both calculations (for C=0 and C=1)
        #evidence = 1 # p(x)
        
        # calculating naive bayes
        c0_calc = prior0 * likelihood0 #/ evidence
        c1_calc = prior1 * likelihood1 #/ evidence
        print(f"likelihoods: {likelihood0} {likelihood1}")
        
        # obtaining predicted class
        output = "" # reset current record output just in case
        if c0_calc > c1_calc:
            output = "0\n"
        elif c0_calc < c1_calc:
            output = "1\n"
        else:
            # Default to inactive on ties (very unlikely occurence)
            tie_count += 1
            output = "0\n"
        
        output_string += output
        
        # END TESTING FILE READ
    
    # Analysis prints
    print(f"tie_count = {tie_count}\nset_null_count = {set_null_count}")

    # Write output to designated file
    out_file.write(output_string)
    # close test and output files
    test_file.close()
    out_file.close()





# *** Neural Network Implementation ***

"""
This train_nn() is the training sequence for the neural network approach
In this perceptron model, 
"""
def train_nn():
    # open training file
    train_file = open(training_file, "r", encoding="utf-8")
    
    
    
    # close train files
    train_file.close()

def test_nn():
    # open testing file
    test_file = open(testing_file, "r", encoding="utf-8")
    out_file = open(output_file, "w")
    output_string = ""
    
    
    
    # Write output to designated file
    #out_file.write(output_string)
    # close test and output files
    test_file.close()
    out_file.close()


# Train and Test Function Execution Code Segment
print("Starting code execution")
train_nb()
test_nb()

"""
train_nn()
test_nn()
"""